import os, time, json, pickle, argparse
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[_v]="6"
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import warnings; warnings.filterwarnings("ignore")
import torch, torch.nn as nn, torch.nn.functional as F
from pytorch_metric_learning.losses import ArcFaceLoss, SubCenterArcFaceLoss, SupConLoss
torch.set_num_threads(6)
SEED=42; CUR=["crime","hate","misinfo","privacy","sexual","violence"]
CURATED=os.environ.get("CARD_WORK", "outputs") + "/curated_benchmark_v2"
DEV="cuda" if torch.cuda.is_available() else "cpu"

def load_curated(tag):
    npy=f"hidden_states_{tag}_curated_v2.npy"; mpk=f"hidden_states_{tag}_meta.pkl"
    H=np.nan_to_num(np.load(os.path.join(CURATED,npy)).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    meta=pickle.load(open(os.path.join(CURATED,mpk),"rb"))["meta"]
    yc=np.array([m["meta_category"] for m in meta]); um=np.isin(yc,CUR)
    return H[um], LabelEncoder().fit(CUR).transform(yc[um])

def best_layer_by_cv(H_u,y,cv):
    L=H_u.shape[1]; tr=cv[0][0]; best=(0,0.)
    for l in range(0,L,max(1,L//12)):
        Xl=H_u[tr,l,:]; a=[]
        for c in range(len(set(y))):
            yy=(y[tr]==c).astype(int); v=Xl[yy==1].mean(0)-Xl[yy==0].mean(0); v/=np.linalg.norm(v)+1e-8
            au=roc_auc_score(yy,Xl@v); a.append(max(au,1-au))
        if np.mean(a)>best[1]: best=(l,np.mean(a))
    return best[0]

class Proj(nn.Module):
    def __init__(s, d_in, dims, emb):
        super().__init__(); layers=[]; prev=d_in
        for h in dims: layers+=[nn.Linear(prev,h),nn.BatchNorm1d(h),nn.ReLU(),nn.Dropout(0.2)]; prev=h
        layers+=[nn.Linear(prev,emb),nn.BatchNorm1d(emb)]; s.net=nn.Sequential(*layers)
    def forward(s,x): return F.normalize(s.net(x),dim=1)

def n_params(*mods): return sum(p.numel() for mm in mods for p in mm.parameters() if p.requires_grad)

def train_torch(Xtr,ytr,Xte,yte,K,loss_name,dims,emb,s=20.,m_deg=10.,sub=3,ls=0.05,epochs=300,lr=2e-3,wd=1e-4,pat=35):
    torch.manual_seed(SEED)
    rng=np.random.RandomState(SEED); perm=rng.permutation(len(Xtr)); nv=max(K*5,len(Xtr)//10); va,tt=perm[:nv],perm[nv:]
    proj=Proj(Xtr.shape[1],dims,emb).to(DEV)
    if loss_name=="ce": head=nn.Linear(emb,K).to(DEV); lf=None; extra=head
    elif loss_name=="arcface": lf=ArcFaceLoss(num_classes=K,embedding_size=emb,margin=m_deg,scale=s).to(DEV); extra=lf; head=None
    elif loss_name=="subcenter": lf=SubCenterArcFaceLoss(num_classes=K,embedding_size=emb,margin=m_deg,scale=s,sub_centers=sub).to(DEV); extra=lf; head=None
    elif loss_name=="supcon": lf=SupConLoss(temperature=0.1); extra=None; head=None
    params=list(proj.parameters())+ (list(extra.parameters()) if extra is not None and hasattr(extra,"parameters") else [])
    opt=torch.optim.AdamW(params,lr=lr,weight_decay=wd)
    cw=torch.tensor([len(tt)/(K*max(1,(ytr[tt]==c).sum())) for c in range(K)],dtype=torch.float32,device=DEV)
    Xt=torch.tensor(Xtr[tt],device=DEV); yt=torch.tensor(ytr[tt],device=DEV)
    Xv=torch.tensor(Xtr[va],device=DEV); yv=ytr[va]
    def logits(emb_):
        if loss_name=="ce": return head(emb_)
        if loss_name in ("arcface","subcenter"): return lf.get_logits(emb_)
        return None
    best=(-1,None,0)
    for ep in range(epochs):
        proj.train();
        if head is not None: head.train()
        opt.zero_grad(); e=proj(Xt)
        if loss_name=="ce": loss=F.cross_entropy(logits(e),yt,weight=cw,label_smoothing=ls)
        else: loss=lf(e,yt)
        loss.backward(); opt.step()
        proj.eval()
        with torch.no_grad():
            ev=proj(Xv)
            if loss_name=="supcon":
                et=proj(Xt).cpu().numpy(); clf=LogisticRegression(C=1,max_iter=2000,class_weight="balanced").fit(et,ytr[tt])
                pv=clf.predict(ev.cpu().numpy())
            else: pv=logits(ev).argmax(1).cpu().numpy()
        f=f1_score(yv,pv,average="macro")
        if f>best[0]+1e-4: best=(f,{ "proj":{k:v.clone() for k,v in proj.state_dict().items()} },ep)
        elif ep-best[2]>pat: break
    proj.load_state_dict(best[1]["proj"]); proj.eval()
    with torch.no_grad():
        ete=proj(torch.tensor(Xte,device=DEV))
        if loss_name=="supcon":
            etr=proj(torch.tensor(Xtr,device=DEV)).cpu().numpy(); clf=LogisticRegression(C=1,max_iter=2000,class_weight="balanced").fit(etr,ytr)
            pred=clf.predict(ete.cpu().numpy())
        else: pred=logits(ete).argmax(1).cpu().numpy()
    return pred, n_params(proj,*( [extra] if (extra is not None and hasattr(extra,"parameters")) else []))

def run_config(H_u,y,layer,cv,name,kind,**kw):
    accs=[];f1s=[];pars=[];t0=time.time()
    for tr,te in cv:
        Xl=H_u[:,layer,:]
        pca=PCA(n_components=min(256,Xl[tr].shape[1],len(tr)-1),whiten=True,random_state=SEED).fit(Xl[tr])
        Xtr=pca.transform(Xl[tr]).astype(np.float32); Xte=pca.transform(Xl[te]).astype(np.float32)
        sc=StandardScaler().fit(Xtr); Xtr,Xte=sc.transform(Xtr).astype(np.float32),sc.transform(Xte).astype(np.float32)
        K=len(set(y))
        if kind=="ldasvc":
            lda=LinearDiscriminantAnalysis(n_components=min(5,K-1)).fit(Xtr,y[tr]); clf=LinearSVC(C=0.1,max_iter=3000,class_weight="balanced").fit(lda.transform(Xtr),y[tr]); pred=clf.predict(lda.transform(Xte)); pc=Xtr.shape[1]*K
        elif kind=="mlp":
            clf=MLPClassifier(kw.get("dims",(256,)),max_iter=300,random_state=SEED).fit(Xtr,y[tr]); pred=clf.predict(Xte); pc=sum(c.size for c in clf.coefs_)+sum(c.size for c in clf.intercepts_)
        else:
            pred,pc=train_torch(Xtr,y[tr],Xte,y[te],K,kind,kw.get("dims",[256]),kw.get("emb",128),s=kw.get("s",20.),m_deg=kw.get("m",10.),sub=kw.get("sub",3))
        accs.append(accuracy_score(y[te],pred)); f1s.append(f1_score(y[te],pred,average="macro")); pars.append(pc)
    return {"name":name,"acc":float(np.mean(accs)),"macroF1":float(np.mean(f1s)),"params":int(np.mean(pars)),"sec":round(time.time()-t0,1)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--tags",nargs="*",default=["qwen3vl_8b"]); args=ap.parse_args()
    GRID=[("LDA+LinSVC(bar)","ldasvc",{}),("MLP-256","mlp",{"dims":(256,)}),
          ("CE+LS p256-128","ce",{"dims":[256],"emb":128}),
          ("ArcFace p256-128","arcface",{"dims":[256],"emb":128,"s":20,"m":10}),
          ("SubCenter p256-128","subcenter",{"dims":[256],"emb":128,"s":20,"m":10,"sub":3}),
          ("SupCon p256-128","supcon",{"dims":[256],"emb":128}),
          ("SubC linear(emb=in)","subcenter",{"dims":[],"emb":128,"s":20,"m":10,"sub":3}),
          ("SubC p128","subcenter",{"dims":[128],"emb":128,"s":20,"m":10,"sub":3}),
          ("SubC p512-256-128","subcenter",{"dims":[512,256],"emb":128,"s":20,"m":10,"sub":3})]
    ALL={}
    for tag in args.tags:
        H_u,y=load_curated(tag); cv=list(StratifiedKFold(5,shuffle=True,random_state=SEED).split(np.zeros(len(y)),y))
        layer=best_layer_by_cv(H_u,y,cv)
        print(f"\n===== {tag} (L={H_u.shape[1]}, single-layer feature @L{layer}={layer/(H_u.shape[1]-1)*100:.0f}%) =====",flush=True)
        print(f"{'config':<22}{'macroF1':>9}{'acc':>8}{'params':>10}{'sec':>7}",flush=True)
        res=[]
        for name,kind,kw in GRID:
            r=run_config(H_u,y,layer,cv,name,kind,**kw); res.append(r)
            print(f"{name:<22}{r['macroF1']*100:>8.1f}{r['acc']*100:>8.1f}{r['params']:>10}{r['sec']:>7}",flush=True)
        ALL[tag]={"layer":int(layer),"results":res}
    json.dump(ALL,open(os.path.join(CURATED,"card_method_search.json"),"w"),indent=2)
    print("\nSaved card_method_search.json",flush=True)

if __name__=="__main__": main()
