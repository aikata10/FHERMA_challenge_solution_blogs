# Matrix Multiplication #

*The article details the solution provided by the winner of the Matrix Multiplication Challenge.*

#### Author: [Aikata (Ph.D. student, TU Graz)](https://www.iaik.tugraz.at/person/aikata-aikata/)

___


<p style="text-align: center;"><img src="https://hackmd.io/_uploads/Skv260T2T.jpg" width="650" /></p>



## Why do we need secure Matrix-Multiplication <img src="https://hackmd.io/_uploads/SJBg0P3hp.png" alt="drawing" width="20"/>

Matrix multiplication is a crucial aspect of advanced mathematics and plays a central role in machine learning, especially in Neural Networks (NN). Certain network components, like fully connected layers or filter/kernel, rely on efficient matrix multiplication. Though there are efficient algorithms like Strassen's algorithm for plaintext operations, conducting matrix multiplication in an encrypted domain is a newer research area. This has gained attention because it enables encrypted ML training or inference using FHE schemes like CKKS, which can handle approximate arithmetic.


## <img src="https://hackmd.io/_uploads/H1n1Q0pnT.png" alt="drawing" width="60"/> Secure Matrix-Multiplication techniques 




Encrypted matrix multiplication techniques in literature can be broadly categorized into three types. The first type involves a depth of two multiplications and utilizes a simple row-wise encoding. An example is the work cited as [[RT22]], which presents a general technique. For a square matrix (dimensions $d\times d$), this method requires $2 d+3\log_2(d)-2$ rotations and $2d$ multiplications. It is important to note that these two operations are the most costly, as an expensive key-switch operation is needed after each one to ensure the ciphertext remains decryptable using the same secret key. A drawback of this approach is the necessity for $d^3$ slots packing availability in the ciphertext. Hence, it does not scale well for big matrices.

The second category of techniques, as described in [[JKLS18]], deviates slightly from previous methods by using diagonal-based matrix multiplication. Although this technique requires $3d+5\sqrt{d}$ rotations and $d$ ciphertext-ciphertext (ct-ct) multiplications initially, with increased packing ($d^3$ slots), it can be optimized to only need $d+2\sqrt{d}$ computations. However, a significant drawback is the requirement for three multiplicative depths. While the algorithms proposed in [[JKLS18]] can be modified to operate within a multiplicative depth of two, doing so results in a much more significant increase in the number of rotations and multiplications.


The third and final category of works, as presented in [[JLK+22],[HZ23],[ZHCJ23]], significantly deviates from the previous two types. These works aim to leverage a multivariate variant of the CKKS scheme (m-RLWE) [[CKY18]], enabling the encoding of a matrix into a hypercube structure. This approach facilitates cost-effective row-wise and column-wise rotations, optimizing matrix multiplication while only requiring a multiplicative depth of two. However, note that the multivariate CKKS is incompatible with the original CKKS.  Additionally, the parameters of the multivariate CKKS need to be standardized, and its initial proposal [[PTP15]] was found to be insecure [[BCV18]].


## <img src="https://hackmd.io/_uploads/SkcCXdh2p.png" alt="drawing" width="65"/> &nbsp; The FHERMA Challenge Solution


<!--- **Algorithm 1** Matrix.Mult (1 Matrix packing version of [[RT22]])

**Require:** $A,B \leftarrow$ row_enc $(\mathtt{A_{d\times d}},\mathtt{B_{d\times d}})$\
**Out:** $C=$ row_enc $(\mathtt{A_{d\times d}}\times\mathtt{B_{d\times d}})$

`// Preprocess A`\
1: **for** $j=0$ to $d-1$ **do**\
2: &nbsp;&nbsp;&nbsp;&nbsp;$\tilde{A}[j] \leftarrow  \texttt{cMult}(A, \pi_{j} )$\
3: &nbsp;&nbsp;&nbsp;&nbsp;$\tilde{A}[j] \leftarrow  \texttt{Rot}(\tilde{A}[j],-j)$\
4: &nbsp;&nbsp;&nbsp;&nbsp;**for** $i=0$ to $\log_2(d)-1$}	**do**\
5: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\tilde{A}[j] +=  \texttt{Rot}(\tilde{A}[j], -(2^i) )$

`// Preprocess B`\
6: **for** $j=0$ to $d-1$ **do**\
7: &nbsp;&nbsp;&nbsp;&nbsp;$\tilde{B}[j] \leftarrow  \texttt{cMult}(B, \psi_{j} )$\
8: &nbsp;&nbsp;&nbsp;&nbsp;$\tilde{B}[j] \leftarrow  \texttt{Rot}(\tilde{B}[j],-j*d)$\
9: &nbsp;&nbsp;&nbsp;&nbsp;**for** $i=0$ to $\log_2(d)-1$ **do**\
10: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\tilde{B}[j] += \texttt{Rot}(\tilde{B}[j], -(2^i)*d )$
 
`// Compute C`\
11: **for** $j=0$ to $d-1$ **do**\
12: &nbsp;&nbsp;&nbsp;&nbsp;$C += \texttt{cMult}(\tilde{A}[j],\tilde{B}[j] )$
--->



Due to the FHERMA challenge restrictions, which limit the multiplicative depth to two and ciphertext encoding to row-wise, the latter two techniques are not applicable. Furthermore, naively applying the first technique [[RT22]] is not feasible as the available slots ($n=8192$) do not meet the requirement of $d^3$ slots for a given dimension of $d=64$. Therefore, we first explore adapting the technique from [[RT22]] to our case. A brief idea of the technique is shown below for matrices of dimension 2.

<p style="text-align: center;">
    <img src="https://hackmd.io/_uploads/HkEdt1J06.png" alt="drawing" width="420"/>
</p>


The adapted technique is outlined in Algorithm 1, which utilizes column and row masks ($\pi_i$ and $\psi_i$, respectively). The complexity of this adaptation is $2d+2d\log_2{d}-2$ rotations and $d$ ct-ct multiplications. Since it is possible to pack two matrices into one ciphertext, the algorithm can be optimized to consume $2d+d\log_2{d}-1$ rotations. This optimization is achieved by aligning rotations at Steps 3 and 8 and then packing two ciphertexts.

<p style="text-align: center;">
    <img src="https://hackmd.io/_uploads/r1OANchnT.png" alt="drawing" width="620"/>
</p>

<img src="https://hackmd.io/_uploads/HJTLXC636.png"  alt="drawing" width="25"/> To optimize this further, we explore a strategy involving packing a duplicate of the initial ciphertext (one original and one rotated) as shown below for d=2.

<p style="text-align: center;">
    <img src="https://hackmd.io/_uploads/S1f4FJJA6.png" alt="drawing" width="620"/>
</p>


This optimization leads to a significant reduction in the number of rotations. The proposed Algorithm 2 outlines this approach, where ciphertexts $A$ and $B$ undergo pre-processing. Following this packing strategy, only $d+d\log_2{d}+1$ rotations and $\frac{d}{2}$ multiplications <img src="https://hackmd.io/_uploads/BJx3CY_23p.png" alt="drawing" width="35"/> are required for the subsequent steps.

<!---
**Algorithm 2** Proposed.Matrix.Mult\
**Require:** $A,B \leftarrow$ row_enc $(\mathtt{A_{d\times d}},\mathtt{B_{d\times d}})$\
**Out:** $C=$ row_enc $(\mathtt{A_{d\times d}}\times\mathtt{B_{d\times d}})$

`// Preprocess A`\
1: $A +=  \texttt{Rot}(A, -d*d+1 )$\
2: **for** $j=0$ to $(d/2)-1$} **do**\
3: &nbsp;&nbsp;&nbsp;&nbsp;$\tilde{A}[j] \leftarrow  \texttt{cMult}(A, \pi_{j,j+d*d} )$\
4: &nbsp;&nbsp;&nbsp;&nbsp;$\tilde{A}[j] \leftarrow  \texttt{Rot}(\tilde{A}[j],-2j)$\
5: &nbsp;&nbsp;&nbsp;&nbsp;**for** $i=0$ to $\log_2(d)-1$ **do**\
6: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\tilde{A}[j] +=  \texttt{Rot}(\tilde{A}[j], -(2^i) )$

`// Preprocess B`\
7: $B +=  \texttt{Rot}(B, -d*d+d )$\
8: **for** $j=0$ to $(d/2)-1$ **do**\
9: &nbsp;&nbsp;&nbsp;&nbsp;$\tilde{B}[j] \leftarrow  \texttt{cMult}(B, \psi_{j,j+d*d} )$

10: &nbsp;&nbsp;&nbsp;&nbsp;$\tilde{B}[j] \leftarrow  \texttt{Rot}(\tilde{B}[j],-2j*d)$\
11: &nbsp;&nbsp;&nbsp;&nbsp;**for** $i=0$ to $\log_2(d)-1$ **do**\
12: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\tilde{B}[j] +=  \texttt{Rot}(\tilde{B}[j], -(2^i)*d )$

`// Compute C`\
13: **for** $j=0$ to $(d/2)-1$ **do**\
14: &nbsp;&nbsp;&nbsp;&nbsp;$C += \texttt{cMult}(\tilde{A}[j],\tilde{B}[j] )$\
15: $C += \texttt{Rot}(c,d* d )$
--->

<p style="text-align: center;">
    <img src="https://hackmd.io/_uploads/SkP-S5226.png" alt="drawing" width="620"/>
</p>


<img src="https://hackmd.io/_uploads/HJTLXC636.png"  alt="drawing" width="25"/>  Additionally, this algorithm exhibits high parallelizability. Hence, employing \'*pragma omp parallel*\' before the for loops can help leverage the capabilities of modern multithreaded operating systems, leading to excellent performance. It is important to highlight that the proposed approach is tailored to the constraints of the FHERMA challenge, considering the available packing $2d^2$ for matrix dimension $d$. Notably, the scalability of this approach improves with higher packing availability. It outperforms all existing methods  [[RT22],[JKLS18]] for $d^3$ slot-packing by consuming only $\mathcal{O}(\log_2{d})$ <img src="https://hackmd.io/_uploads/ryckn_n3a.png" alt="drawing" width="45"/> rotations while still requiring a multiplicative depth of two. The proposed algorithm is characterized by its simplicity, yet it achieves non-trivial results, particularly when applied to higher packing scenarios. Its effectiveness lies in improving the best-case outcomes.

Extending and applying the proposed approach to rectangular matrices and scenarios involving smaller matrix filters applied to a matrix present intriguing possibilities for future exploration. Generalizing the algorithm to handle different matrix shapes and sizes could enhance its versatility and applicability in various contexts within the field of privacy-preserving matrix-multiplication.


___
## <img src="https://hackmd.io/_uploads/BknuCw_pT.png"  alt="drawing" width="30"/>  Take the Red Pill!

Dive into the private Matrix Multiplication implementation live at: https://github.com/Fherma-challenges/matrix-mult
___
[RT22]: https://doi.org/10.1145/3560810.3564267
[JKLS18]: https://eprint.iacr.org/2018/1041.pdf
[JLK+22]: https://doi.org/10.1145/3488932.3523253
[HZ23]: https://doi.org/10.1007/s11227-022-04850-4
[ZHCJ23]: https://doi.org/10.1007/978-3-031-50594-2\_13
[CKY18]: https://eprint.iacr.org/2018/1245
[PTP15]: https://doi.org/10.1109/ICASSP.2015.7178262
[BCV18]: https://eprint.iacr.org/2018/966

<p style="text-align:center;font-family:georgia,garamond;">  Made with ♥️ </p>
