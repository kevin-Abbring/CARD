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
from sklearn.metrics import roc_auc_score, confusion_matrix, silhouette_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings; warnings.filterwarnings("ignore")
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
def best_layer(H,yc):
    um=np.isin(yc,CLS); sm=(yc=="safe"); yb=um[um|sm].astype(int); best=(0,0.)
    for l in range(H.shape[1]):
        Xl=H[:,l,:]; v=Xl[um].mean(0)-Xl[sm].mean(0); v/=np.linalg.norm(v)+1e-8
        a=roc_auc_score(yb,Xl[um|sm]@v); a=max(a,1-a)
        if a>best[1]: best=(l,a)
    return best[0]

def card_feat(H,yc,bl):
    return H[np.isin(yc,CLS),bl,:].astype(np.float32)

def sixway_oof(X,y,head="card"):
    pred=np.zeros(len(X),int)
    for tr,te in cvf(y):
        if head=="card":
            pca=PCA(n_components=min(256,X.shape[1],len(tr)-1),whiten=True,random_state=SEED).fit(X[tr])
            a,b=pca.transform(X[tr]),pca.transform(X[te]); sc=StandardScaler().fit(a)
            clf=LogisticRegression(C=1,max_iter=3000,class_weight="balanced").fit(sc.transform(a),y[tr]); pred[te]=clf.predict(sc.transform(b))
        else:
            sc=StandardScaler().fit(X[tr]); lda=LinearDiscriminantAnalysis(n_components=5).fit(sc.transform(X[tr]),y[tr])
            svm=LinearSVC(C=0.1,max_iter=3000,class_weight="balanced").fit(lda.transform(sc.transform(X[tr])),y[tr]); pred[te]=svm.predict(lda.transform(sc.transform(X[te])))
    return pred
def ovr_auroc(X,y,head="card"):
    per={}
    for cid,c in enumerate(CLS):
        yb=(y==cid).astype(int); a=[]
        for tr,te in cvf(yb):
            if head=="card":
                pca=PCA(n_components=min(256,X.shape[1],len(tr)-1),whiten=True,random_state=SEED).fit(X[tr]); A,B=pca.transform(X[tr]),pca.transform(X[te])
            else: A,B=X[tr],X[te]
            sc=StandardScaler().fit(A); lr=LogisticRegression(C=1,max_iter=2000,class_weight="balanced").fit(sc.transform(A),yb[tr])
            a.append(roc_auc_score(yb[te],lr.predict_proba(sc.transform(B))[:,1]))
        per[c]=float(np.mean(a))
    return per
def pairwise_matrix(Xc,y,nested=False):
    M=np.full((NC,NC),np.nan)
    for i,j in combinations(range(NC),2):
        m=(y==i)|(y==j); Xs=Xc[m]; ys=(y[m]==j).astype(int); aus=[]
        for tr,te in cvf(ys):
            k=min(150,Xs[tr].shape[0]-1,Xs[tr].shape[1]); sv=TruncatedSVD(k,random_state=SEED).fit(Xs[tr])
            A,B=sv.transform(Xs[tr]),sv.transform(Xs[te]); sc=StandardScaler().fit(A)
            lda=LinearDiscriminantAnalysis(n_components=1).fit(sc.transform(A),ys[tr]); pr=lda.transform(sc.transform(B)).ravel()
            au=roc_auc_score(ys[te],pr); aus.append(max(au,1-au))
        M[i,j]=M[j,i]=np.mean(aus)
    return M

RES={}
data_fig={}
for tag,name in TAGS:
    if not os.path.exists(f"{OUT}/hidden_states_{tag}_curated_v2.npy"): print(f"[skip]{tag}"); continue
    H,yc=load(tag); um=np.isin(yc,CLS); y=LabelEncoder().fit(CLS).transform(yc[um]); bl=best_layer(H,yc)
    Xc=card_feat(H,yc,bl)
    feat=refusal_feat(H,yc); fu=feat[um]
    pred_c=sixway_oof(Xc,y,"card"); pred_h=sixway_oof(fu,y,"hd")
    pc_card={c:float((pred_c[y==i]==i).mean()) for i,c in enumerate(CLS)}; pc_hd={c:float((pred_h[y==i]==i).mean()) for i,c in enumerate(CLS)}
    auc_card=ovr_auroc(Xc,y,"card"); auc_hd=ovr_auroc(fu,y,"hd")
    pw=pairwise_matrix(Xc,y)
    RES[tag]={"name":name,"best_layer":int(bl),"acc_card":float((pred_c==y).mean()),"acc_hd":float((pred_h==y).mean()),
              "per_cat_6way_card":pc_card,"per_cat_6way_hd":pc_hd,"ovr_auroc_card":auc_card,"ovr_auroc_hd":auc_hd,
              "pairwise_mean":float(np.nanmean(pw)),"pairwise_min":float(np.nanmin(pw)),
              "pairwise":{f"{CLS[i]}_{CLS[j]}":float(pw[i,j]) for i,j in combinations(range(NC),2)}}
    data_fig[tag]={"Xc":Xc,"y":y,"name":name,"pred_c":pred_c,"pred_h":pred_h,"pw":pw}
    print(f"{name:<13} CARD 6way={RES[tag]['acc_card']*100:.1f} HD={RES[tag]['acc_hd']*100:.1f} pairwise mean={RES[tag]['pairwise_mean']:.3f} min={RES[tag]['pairwise_min']:.3f} L{bl}",flush=True)

if "qwen3vl_8b" in data_fig:
    d=data_fig["qwen3vl_8b"]
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    for ax,(pred,ttl) in zip(axs,[(d["pred_c"],"CARD (strengthened head)"),(d["pred_h"],"HiddenDetect")]):
        cm=confusion_matrix(d["y"],pred); cmn=cm/cm.sum(1,keepdims=True)
        im=ax.imshow(cmn,cmap="Blues",vmin=0,vmax=1)
        ax.set_xticks(range(NC)); ax.set_yticks(range(NC)); ax.set_xticklabels(CLS,rotation=45,ha="right",fontsize=8); ax.set_yticklabels(CLS,fontsize=8)
        for i in range(NC):
            for j in range(NC): ax.text(j,i,cm[i,j],ha="center",va="center",fontsize=7,color="white" if cmn[i,j]>.5 else "black")
        ax.set_title(f"{ttl}  (acc {(pred==d['y']).mean()*100:.1f}%)",fontsize=10); ax.set_xlabel("predicted"); ax.set_ylabel("true")
    plt.suptitle("6-way confusion matrices (Qwen3-VL-8B, curated 5-fold OOF)",fontweight="bold"); plt.tight_layout()
    plt.savefig(f"{FIG}/confusion_heatmap.png",dpi=150,bbox_inches="tight"); plt.close(); print("saved confusion_heatmap.png",flush=True)
if "qwen3vl_8b" in data_fig:
    d=data_fig["qwen3vl_8b"]; pw=d["pw"]; Xc=d["Xc"]; y=d["y"]
    pairs=sorted(combinations(range(NC),2),key=lambda ij:-pw[ij[0],ij[1]])[:6]
    fig,axs=plt.subplots(2,3,figsize=(13,6.5)); axs=axs.flatten(); col=plt.get_cmap("tab10")
    for ax,(i,j) in zip(axs,pairs):
        m=(y==i)|(y==j); Xs=Xc[m]; ys=(y[m]==j).astype(int)
        k=min(150,Xs.shape[0]-1,Xs.shape[1]); sv=TruncatedSVD(k,random_state=SEED).fit(Xs); A=sv.transform(Xs); sc=StandardScaler().fit(A)
        lda=LinearDiscriminantAnalysis(n_components=1).fit(sc.transform(A),ys); pr=lda.transform(sc.transform(A)).ravel()
        ax.hist(pr[ys==0],bins=25,alpha=.55,color=col(i),label=CLS[i]); ax.hist(pr[ys==1],bins=25,alpha=.55,color=col(j),label=CLS[j])
        ax.set_title(f"{CLS[i]} vs {CLS[j]}  AUROC={pw[i,j]:.3f}",fontsize=9); ax.legend(fontsize=7); ax.set_xlabel("LDA-1D")
    plt.suptitle("Pairwise 1-D separability (Qwen3-VL-8B)",fontweight="bold"); plt.tight_layout()
    plt.savefig(f"{FIG}/pairwise_separation.png",dpi=150,bbox_inches="tight"); plt.close(); print("saved pairwise_separation.png",flush=True)
order=[t for t,_ in TAGS if t in data_fig][:4]
if len(order)>=4:
    fig=plt.figure(figsize=(15,13)); col=plt.get_cmap("tab10")
    for idx,tag in enumerate(order):
        d=data_fig[tag]; Xc=d["Xc"]; y=d["y"]
        sc=StandardScaler().fit(Xc); sv=TruncatedSVD(min(50,Xc.shape[1]),random_state=SEED).fit(sc.transform(Xc))
        lda=LinearDiscriminantAnalysis(n_components=5).fit(sv.transform(sc.transform(Xc)),y); P=lda.transform(sv.transform(sc.transform(Xc)))
        sil=silhouette_score(P[:,:3],y); ax=fig.add_subplot(2,2,idx+1,projection="3d")
        for cid,c in enumerate(CLS):
            m=y==cid; ax.scatter(P[m,0],P[m,1],P[m,2],c=[col(cid)],s=9,alpha=.5,edgecolors="white",linewidths=.12,label=c)
        ax.set_title(f"{d['name']}  silhouette={sil:.3f}",fontsize=10.5); ax.view_init(elev=18,azim=-65)
        if idx==0: ax.legend(fontsize=8,loc="upper left")
    plt.suptitle("3-D LDA projection of curated unsafe samples (per backbone)",fontweight="bold",y=.995); plt.tight_layout(rect=[0,0,1,.975])
    plt.savefig(f"{FIG}/lda_3d_universality.png",dpi=150,bbox_inches="tight"); plt.close(); print("saved lda_3d_universality.png",flush=True)

json.dump(RES,open(f"{OUT}/card_full_v2.json","w"),indent=2); print("\nSaved card_full_v2.json",flush=True)
