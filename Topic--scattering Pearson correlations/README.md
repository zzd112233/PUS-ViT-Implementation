# Topic–scattering correlation

This folder reproduces the Pearson correlations between learned topic activations and polarimetric features in Figs. 11 and 16. Both figures come from the checkpoints and paired validation lists stored here. Topic index \(T_k\) is the \(k\)-th component of that activation. It is not an entry of the coherency matrix.

## Sample selection

Each dataset uses the paired validation lists from Table III of the manuscript. `T_validation.txt` and `RSD_validation.txt` are line-aligned: line \(i\) is the same patch in the coherency-matrix list and the 12-channel RSD list. These are validation samples, not the remaining labeled test pixels used for final classification evaluation.

| Dataset | Patches | Per class | Labels |
| --- | ---: | ---: | --- |
| Barnaul (Fig. 16) | 6,948 | 772 | 0–8 |
| Oberpfaffenhofen (Fig. 11) | 4,368 | 624 | 0–6 |

Background pixels are not in these lists. The polarimetric vector of a patch is the spatial mean of its RSD channels, in this order: Span, \(P_S\), \(P_D\), \(P_V\), \(P_2\), \(P_3\), \(\theta\), \(\phi\), \(x\), \(y\), \(a\), \(b\).

## Topic activation and checkpoints

The topic vector is the softmax of the last linear layer in `feature_to_topic`, taken before the ReLU used by PGCL. The correlation is the Pearson coefficient between that component and one polarimetric feature over the paired validation patches.

| Dataset | Stage-2 fusion checkpoint | Stage-1 topic checkpoint |
| --- | --- | --- |
| Barnaul | `Barnaul/model/fusion_model_best.pth` | `Barnaul/model/pgcl_model_best.pth` |
| Oberpfaffenhofen | `Ober/model/fusion_model_best.pth` | `Ober/model/pgcl_model_best.pth` |

The saved matrices are `Barnaul/metric1_topic_scattering_corr.csv` and `Ober/topic_quantity_corr.csv`. Cells with absolute value at least 0.12 are annotated to two decimals in the PDF, which is why \(r(T_6,P_D)=-0.4159\) appears as \(-0.42\).

Checked Barnaul cells: \(r(T_6,P_D)=-0.42\), \(r(T_{13},P_V)=0.14\), \(r(T_{14},P_2)=0.60\), \(r(T_{12},\mathrm{Span})=0.26\). The value \(-0.30\) in the \(T_{13}\) row is \(P_2\), not \(P_V\).

Checked Oberpfaffenhofen cells: \(r(T_1,P_S)=-0.77\), \(r(T_{12},\mathrm{Span})=0.70\), \(r(T_7,P_D)=0.69\).

These associations describe mixtures of polarimetric features. They do not assign each topic to one scattering mechanism.

## Run

From `code/`, with the patch files named in the two validation lists available on disk:

```text
python Plot_Five_Metrics_PDF.py
python Plot_Ober_Three_Metrics.py
```

`Plot_Five_Metrics_PDF.py` redraws the Barnaul correlation as its first figure and writes `Barnaul/metric1_topic_scattering_corr.csv`. `Plot_Ober_Three_Metrics.py` does the same for Oberpfaffenhofen and writes `Ober/topic_quantity_corr.csv`.
