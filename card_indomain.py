import os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[v]="8"
import json, pickle, numpy as np
from itertools import combinations
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, silhouette_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings; warnings.filterwarnings("ignore")
import torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(8)
OUT=os.environ.get("CARD_WORK", "outputs") + "/curated_benchmark_v2"
FIG=os.environ.get("CARD_WORK", "outputs") + "/figures"; os.makedirs(FIG,exist_ok=True)
SEED=42; CLS=["crime","hate","misinfo","privacy","sexual","violence"]; NC=6
TAGS=[("qwen3vl_4b","Qwen3-VL-4B"),("qwen3vl_8b","Qwen3-VL-8B"),("gemma3_12b","Gemma-3-12B"),("llava15_7b","LLaVA-1.5-7B")]
def cvf(y): return list(StratifiedKFold(5,shuffle=True,random_state=SEED).split(np.zeros(len(y)),y))
def load(tag):
    H=np.nan_to_num(np.load(f"{OUT}/hidden_states_{tag}_curated_v2.npy").astype(np.float32),posinf=0,neginf=0)
    meta=pickle.load(open(f"{OUT}/hidden_states_{tag}_meta.pkl","rb"))["meta"]
    yc=np.array([m["meta_category"] for m in meta]); return H,yc
def refusal_feat(H,yc):
    N,L,D=H.shape; um=np.isin(yc,CLS); sm=(yc=="safe"); f=np.zeros((N,L),np.float32)
    for l in range(L):
        Xl=H[:,l,:]; v=Xl[um].mean(0)-Xl[sm].mean(0); v/=np.linalg.norm(v)+1e-8
        Xn=Xl/(np.linalg.norm(Xl,axis=1,keepdims=True)+1e-8); f[:,l]=Xn@v
    return f

class MLP(nn.Module):
    def __init__(s,d,emb=128,p=0.3):
        super().__init__(); s.net=nn.Sequential(nn.Linear(d,256),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(p),
                                                nn.Linear(256,emb),nn.BatchNorm1d(emb),nn.ReLU(),nn.Dropout(p)); s.cls=nn.Linear(emb,NC)
    def emb(s,x): return s.net(x)
    def forward(s,x): return s.cls(s.net(x))
def train_mlp(Xtr,ytr,Xva,yva,epochs=300,lr=2e-3,wd=1e-4,ls=0.05,pat=35):
    torch.manual_seed(SEED); m=MLP(Xtr.shape[1]); opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=wd)
    cw=torch.tensor([len(ytr)/(NC*max(1,(ytr==c).sum())) for c in range(NC)],dtype=torch.float32)
    Xt=torch.tensor(Xtr);yt=torch.tensor(ytr);Xv=torch.tensor(Xva);yv=torch.tensor(yva); best=(1e9,None,0)
    for ep in range(epochs):
        m.train();opt.zero_grad(); F.cross_entropy(m(Xt),yt,weight=cw,label_smoothing=ls).backward(); opt.step()
        m.eval()
        with torch.no_grad(): vl=F.cross_entropy(m(Xv),yv).item()
        if vl<best[0]-1e-4: best=(vl,{k:v.clone() for k,v in m.state_dict().items()},ep)
        elif ep-best[2]>pat: break
    m.load_state_dict(best[1]); return m

def pca_layer(H_u,l,tr,te):
    Xl=H_u[:,l,:]; pca=PCA(n_components=min(256,Xl.shape[1],len(tr)-1),whiten=True,random_state=SEED).fit(Xl[tr])
    a,b=pca.transform(Xl[tr]).astype(np.float32),pca.transform(Xl[te]).astype(np.float32)
    sc=StandardScaler().fit(a); return sc.transform(a).astype(np.float32),sc.transform(b).astype(np.float32)

def select_layer(H_u,y,layers):
    best=(layers[0],0.)
    for l in layers:
        acc=[]
        for tr,te in cvf(y):
            a,b=pca_layer(H_u,l,tr,te); lr=LogisticRegression(C=1,max_iter=2000,class_weight="balanced").fit(a,y[tr]); acc.append(accuracy_score(y[te],lr.predict(b)))
        if np.mean(acc)>best[1]: best=(l,np.mean(acc))
    return best[0]

def mlp_oof(H_u,y,bl):
    pred=np.zeros(len(y),int); accs=[]
    for tr,te in cvf(y):
        a,b=pca_layer(H_u,bl,tr,te)
        rng=np.random.RandomState(SEED); perm=rng.permutation(len(a)); nv=max(NC*8,len(a)//10); va,tt=perm[:nv],perm[nv:]
        m=train_mlp(a[tt],y[tr][tt],a[va],y[tr][va])
        m.eval()
        with torch.no_grad(): pr=m(torch.tensor(b)).argmax(1).numpy()
        pred[te]=pr; accs.append((pr==y[te]).mean())
    return pred,float(np.mean(accs)),float(np.std(accs))
def hd_oof(fu,y):
    pred=np.zeros(len(y),int)
    for tr,te in cvf(y):
        sc=StandardScaler().fit(fu[tr]); lda=LinearDiscriminantAnalysis(n_components=5).fit(sc.transform(fu[tr]),y[tr])
        svm=LinearSVC(C=0.1,max_iter=3000,class_weight="balanced").fit(lda.transform(sc.transform(fu[tr])),y[tr]); pred[te]=svm.predict(lda.transform(sc.transform(fu[te])))
    return pred
def ovr_auroc(H_u,y,bl):
    per={}
    for cid,c in enumerate(CLS):
        yb=(y==cid).astype(int); a=[]
        for tr,te in cvf(yb):
            A,B=pca_layer(H_u,bl,tr,te); lr=LogisticRegression(C=1,max_iter=2000,class_weight="balanced").fit(A,yb[tr]); a.append(roc_auc_score(yb[te],lr.predict_proba(B)[:,1]))
        per[c]=float(np.mean(a))
    return per
def pairwise(H_u,y,bl):
    M=np.full((NC,NC),np.nan); Xl=H_u[:,bl,:]
    for i,j in combinations(range(NC),2):
        m=(y==i)|(y==j); Xs=Xl[m]; ys=(y[m]==j).astype(int); aus=[]
        for tr,te in cvf(ys):
            k=min(150,Xs[tr].shape[0]-1,Xs[tr].shape[1]); sv=TruncatedSVD(k,random_state=SEED).fit(Xs[tr]); A,B=sv.transform(Xs[tr]),sv.transform(Xs[te]); sc=StandardScaler().fit(A)
            lda=LinearDiscriminantAnalysis(n_components=1).fit(sc.transform(A),ys[tr]); pr=lda.transform(sc.transform(B)).ravel(); au=roc_auc_score(ys[te],pr); aus.append(max(au,1-au))
        M[i,j]=M[j,i]=np.mean(aus)
    return M

RES={}; figd={}
for tag,name in TAGS:
    if not os.path.exists(f"{OUT}/hidden_states_{tag}_curated_v2.npy"): print(f"[skip]{tag}"); continue
    H,yc=load(tag); um=np.isin(yc,CLS); y=LabelEncoder().fit(CLS).transform(yc[um]); H_u=H[um]; L=H.shape[1]
    layers=list(range(0,L,max(1,L//12)))
    bl=select_layer(H_u,y,layers)
    pred_c,acc_c,std_c=mlp_oof(H_u,y,bl); fu=refusal_feat(H,yc)[um]; pred_h=hd_oof(fu,y)
    auc=ovr_auroc(H_u,y,bl); pw=pairwise(H_u,y,bl)
    RES[tag]={"name":name,"layer":int(bl),"rel":round(bl/(L-1),2),"acc_card":acc_c,"std_card":std_c,"acc_hd":float((pred_h==y).mean()),
              "per_cat_card":{c:float((pred_c[y==i]==i).mean()) for i,c in enumerate(CLS)},
              "per_cat_hd":{c:float((pred_h[y==i]==i).mean()) for i,c in enumerate(CLS)},
              "ovr_auroc":auc,"pairwise_mean":float(np.nanmean(pw)),"pairwise_min":float(np.nanmin(pw)),
              "pairwise":{f"{CLS[i]}_{CLS[j]}":float(pw[i,j]) for i,j in combinations(range(NC),2)}}
    figd[tag]={"H_u":H_u,"y":y,"name":name,"pred_c":pred_c,"pred_h":pred_h,"pw":pw,"bl":bl}
    print(f"{name:<13} L{bl}({RES[tag]['rel']}) CARD={acc_c*100:.1f}+/-{std_c*100:.1f} HD={RES[tag]['acc_hd']*100:.1f} pair mean={RES[tag]['pairwise_mean']:.3f} min={RES[tag]['pairwise_min']:.3f}",flush=True)

if "qwen3vl_8b" in figd:
    d=figd["qwen3vl_8b"]; fig,axs=plt.subplots(1,2,figsize=(12,5))
    for ax,(pred,ttl) in zip(axs,[(d["pred_c"],"CARD (strengthened head)"),(d["pred_h"],"HiddenDetect")]):
        cm=confusion_matrix(d["y"],pred); cmn=cm/cm.sum(1,keepdims=True); ax.imshow(cmn,cmap="Blues",vmin=0,vmax=1)
        ax.set_xticks(range(NC));ax.set_yticks(range(NC));ax.set_xticklabels(CLS,rotation=45,ha="right",fontsize=8);ax.set_yticklabels(CLS,fontsize=8)
        for i in range(NC):
            for j in range(NC): ax.text(j,i,cm[i,j],ha="center",va="center",fontsize=7,color="white" if cmn[i,j]>.5 else "black")
        ax.set_title(f"{ttl} (acc {(pred==d['y']).mean()*100:.1f}%)",fontsize=10);ax.set_xlabel("predicted");ax.set_ylabel("true")
    plt.suptitle("6-way confusion (Qwen3-VL-8B, curated 5-fold OOF)",fontweight="bold");plt.tight_layout();plt.savefig(f"{FIG}/confusion_heatmap.png",dpi=150,bbox_inches="tight");plt.close();print("saved confusion",flush=True)
if "qwen3vl_8b" in figd:
    d=figd["qwen3vl_8b"];pw=d["pw"];Xl=d["H_u"][:,d["bl"],:];y=d["y"];col=plt.get_cmap("tab10")
    pairs=sorted(combinations(range(NC),2),key=lambda ij:-pw[ij[0],ij[1]])[:6]
    fig,axs=plt.subplots(2,3,figsize=(13,6.5));axs=axs.flatten()
    for ax,(i,j) in zip(axs,pairs):
        m=(y==i)|(y==j);Xs=Xl[m];ys=(y[m]==j).astype(int);k=min(150,Xs.shape[0]-1,Xs.shape[1]);sv=TruncatedSVD(k,random_state=SEED).fit(Xs);A=sv.transform(Xs);sc=StandardScaler().fit(A)
        lda=LinearDiscriminantAnalysis(n_components=1).fit(sc.transform(A),ys);pr=lda.transform(sc.transform(A)).ravel()
        ax.hist(pr[ys==0],bins=25,alpha=.55,color=col(i),label=CLS[i]);ax.hist(pr[ys==1],bins=25,alpha=.55,color=col(j),label=CLS[j])
        ax.set_title(f"{CLS[i]} vs {CLS[j]} AUROC={pw[i,j]:.3f}",fontsize=9);ax.legend(fontsize=7)
    plt.suptitle("Pairwise 1-D separability (Qwen3-VL-8B)",fontweight="bold");plt.tight_layout();plt.savefig(f"{FIG}/pairwise_separation.png",dpi=150,bbox_inches="tight");plt.close();print("saved pairwise",flush=True)
order=[t for t,_ in TAGS if t in figd][:4]
if len(order)>=4:
    fig=plt.figure(figsize=(15,13));col=plt.get_cmap("tab10")
    for idx,tag in enumerate(order):
        d=figd[tag];Xl=d["H_u"][:,d["bl"],:];y=d["y"];sc=StandardScaler().fit(Xl);sv=TruncatedSVD(min(50,Xl.shape[1]),random_state=SEED).fit(sc.transform(Xl))
        lda=LinearDiscriminantAnalysis(n_components=5).fit(sv.transform(sc.transform(Xl)),y);P=lda.transform(sv.transform(sc.transform(Xl)));sil=silhouette_score(P[:,:3],y)
        ax=fig.add_subplot(2,2,idx+1,projection="3d")
        for cid,c in enumerate(CLS):
            m=y==cid;ax.scatter(P[m,0],P[m,1],P[m,2],c=[col(cid)],s=9,alpha=.5,edgecolors="white",linewidths=.12,label=c)
        ax.set_title(f"{d['name']} silhouette={sil:.3f}",fontsize=10.5);ax.view_init(elev=18,azim=-65)
        if idx==0: ax.legend(fontsize=8,loc="upper left")
    plt.suptitle("3-D LDA projection of curated unsafe samples",fontweight="bold",y=.995);plt.tight_layout(rect=[0,0,1,.975]);plt.savefig(f"{FIG}/lda_3d_universality.png",dpi=150,bbox_inches="tight");plt.close();print("saved lda3d",flush=True)
json.dump(RES,open(f"{OUT}/card_indomain.json","w"),indent=2);print("\nSaved card_indomain.json",flush=True)
