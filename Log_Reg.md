# Logistic Regression Function 

[![hackmd-github-sync-badge](https://hackmd.io/r_dCvgUOR2OThXKFiqnzrg/badge)](https://hackmd.io/r_dCvgUOR2OThXKFiqnzrg)


*The article details the solution provided by the winner of the Logistic Regression Challenge.*

#### Author: [Aikata (Ph.D. student, TU Graz)](https://www.iaik.tugraz.at/person/aikata-aikata/) 
---

## What is Logistic function <img src="https://hackmd.io/_uploads/SJBg0P3hp.png" alt="drawing" width="20" />

<p style="text-align: center;"><img src="https://hackmd.io/_uploads/Hk_Td06ha.png" width="650" /> <br> <small><small> Inspired from <a href="https://www.imaginary.org/gallery/maths-dance-moves">Imaginary</a></small></small>
</p>


The logistic (a.k.a., sigmoid) function- $\frac{1}{1+e^{-x}}$ forms the basis for logistic regression-based machine learning. It is also commonly employed as a non-linear activation function in neural networks. Homomorphic schemes like CKKS only support polynomial arithmetic operations, and expressing the sigmoid function as a polynomial is not feasible. Therefore, various series, such as the Taylor series or the Chebyshev series, are utilized to approximate the sigmoid function. Recent efforts have focused on developing privacy-preserving models for logistic regression training and inference by employing similar approximations of the logistic function [[KSK+18],[KSW+18],[XWBB16],[AHPW16]]. Therefore, it is crucial to investigate the extent to which we can approximate the logistic function under constrained multiplicative depth. 


## <img src="https://hackmd.io/_uploads/SkcCXdh2p.png" alt="drawing" width="65"/> &nbsp; The FHERMA Challenge 


The challenge has two sets of test cases, distinguished solely in terms of the permitted multiplicative depth. The input values for both sets of test cases are confined to the range of [−25, 25]. The straightforward logistic evaluation of this range is illustrated in Figure 1. 


<p float="left">
  <img src="https://hackmd.io/_uploads/BkdAYe83a.png" width="350" />
  <img src="https://hackmd.io/_uploads/HkT6FgLhp.png" width="350" /> 
</p>

<p style="text-align: center;"><b>Figure 1:</b> Sigmoid function and its approximation using Chebyshev series.</p>

One standard method for approximating the logistic function involves employing the Chebyshev series. However, the Chebyshev series provides results over the domain [−1, 1]. Therefore, to apply the Chebyshev series to broader domains (e.g., [a, b]), the input polynomial’s (e.g., $x$) domain is scaled to bring it to the interval [−1, 1] as outlined below, ensuring the applicability of the Chebyshev approximation.


\begin{align}
    x' & = \frac{2x - (b+a)}{b-a} \\
    x' & = \frac{x}{\omega} \hspace{20pt} \text{ when, } |a|=|b|, a \neq b,\text{ and }\omega=|b|
\end{align}

The scaled ciphertext $x'$ is subsequently employed for the approximation. However, note that this scaling results in the loss of one multiplicative depth. This aspect is explicitly mentioned in the [library documentation](https://github.com/openfheorg/openfhe-development/blob/main/src/pke/examples/FUNCTION_EVALUATION.md) for function evaluation. 



Consequently, when presented with a challenge specifying a multiplicative depth $d$ over an arbitrary range $[a,b]$, such that $a\neq-1$ and $b\neq1$, the available Chebyshev series approximation technique allows us to evaluate up to a depth of $d-1$ at best. To fully exploit this using OpenFHE, the technique described in the SIGN challenge solution can be applied. However, it becomes evident that although reducing the depth from 7 to 6 $(2^7=128 \rightarrow 2^6=64),$ as shown in Figure 1 (a) does not result in a significant loss of accuracy, but dropping from 4 to 3 ($2^4 = 16 \rightarrow 2^3 = 8$) entails a considerable loss (Figure 1 (b)). Therefore, our exploration focuses on enhancing this existing technique to yield the best approximation results.

## <img src="https://hackmd.io/_uploads/BJUDuu336.png" alt="drawing" width="50" />  &nbsp;  Test Case 1 

The first test case permits a multiplicative depth of seven. Initially, we assess the capabilities of the current implementation to determine how well we can approximate. Notably, with OpenFHE's ChebyshevPS implementation, we can evaluate the Chebyshev series up to the coefficient 59.  Applying the SIGN challenge strategy can extend this evaluation to 63 coefficients, resulting in an impressive accuracy of $99.99\%$. To walk the extra mile (or 0.01\%) for achieving complete accuracy, we delve into the recursive unroll of $\textbf{T}_{65}$ as outlined below:

 \begin{align}
    \text{c}\times\textbf{T}_{65} && \rightarrow && \text{2c}\times\textbf{T}_{32}\times\textbf{T}_{33}-\text{c}\times\textbf{T}_{1} && \rightarrow &&  \textbf{T}_{32}\times(\text{2c}\times\textbf{T}_{33})-\text{c}\times\textbf{T}_{1}\\
    \text{2c}\times\textbf{T}_{33} && \rightarrow &&  \text{4c}\times\textbf{T}_{16}\times\textbf{T}_{17}-\text{2c}\times\textbf{T}_{1} && \rightarrow  && \textbf{T}_{16}\times(\text{4c}\times\textbf{T}_{17})-\text{2c}\times\textbf{T}_{1}\\
    \text{4c}\times\textbf{T}_{17} && \rightarrow  && \text{8c}\times\textbf{T}_{8}\times\textbf{T}_{9}-\text{4c}\times\textbf{T}_{1} && \rightarrow &&  \textbf{T}_{8}\times(\text{8c}\times\textbf{T}_{9})-\text{4c}\times\textbf{T}_{1}\\
    \text{8c}\times\textbf{T}_{9} && \rightarrow &&  \text{16c}\times\textbf{T}_{4}\times\textbf{T}_{5}-\text{8c}\times\textbf{T}_{1} && \rightarrow &&  \textbf{T}_{4}\times(\text{16c}\times\textbf{T}_{5})-\text{8c}\times\textbf{T}_{1}\\
    \text{16c}\times\textbf{T}_{5} && \rightarrow &&  \text{32c}\times\textbf{T}_{2}\times\textbf{T}_{3}-\text{16c}\times\textbf{T}_{1} && \rightarrow &&  \textbf{T}_{2}\times(\text{32c}\times\textbf{T}_{3})-\text{16c}\times\textbf{T}_{1}
\end{align}

Note that this recursive breakdown is just one of the numerous possible approaches. Examining this breakdown, we notice that at each step, the two coefficients to be multiplied are at different depths, and one consistently exceeds the exploitable multiplicative depth at that level. Thus, we continue the breakdown until we reach $\textbf{T}_2$ and $\textbf{T}_3$, which can be represented as follows:

 \begin{align}
    \textbf{T}_{2} && \rightarrow && \text{2x}'^2-1 && \text{// Multiplicative depth 2}\\
    c\times\textbf{T}_{3} && \rightarrow && \text{4cx}'^3-\text{3cx}'  && \text{// Multiplicative depth 3}
\end{align}

<img src="https://hackmd.io/_uploads/HJTLXC636.png"  alt="drawing" width="25" /> $\textbf{T}_3$ is at a higher depth than $\textbf{T}_2$. The computation of $\textbf{T}_{65}$ relies on our ability to compute $32c\times\textbf{T}_3$ while remaining at the same multiplicative depth as $\textbf{T}_2$. In this context, we emphasize that '*the sky is not the limit*' <img src="https://hackmd.io/_uploads/BJx3CY_23p.png" alt="drawing" width="30"/>  and indeed, there is a method to accomplish this, as illustrated in the equations below:

\begin{align}
     32c\times\textbf{T}_{3} && \rightarrow && \frac{128c}{\omega^3} x^3-\text{96cx}'  && \text{// Multiplicative depth 2}\\
      && \rightarrow && (\frac{128c}{\omega^3}x)(x^2)-\text{96cx}'  && \text{// Multiplicative depth 2}
\end{align}

The logistic function is an odd function, and only the odd coefficients of the Chebyshev series contribute to its approximation. Leveraging recursive breakdowns like the one for $\textbf{T}_{65}$, we calculate odd series coefficients up to $\textbf{T}_{77}$, successfully achieving our target of $100\%$ (99.9988\%) accuracy for this particular test case. While approximating up to  $\textbf{T}_{77}$ sufficed for the first test case, it remains interesting to investigate how far we can extend this breakdown before reaching limitations.

## <img src="https://hackmd.io/_uploads/BJUDuu336.png" alt="drawing" width="50" />  &nbsp;  Test Case 2

In this particular test case, a multiplicative depth of four is mandated. <img src="https://hackmd.io/_uploads/HJjomKn3p.png" alt="drawing" width="45" />

The straight-forward Chebyshev series computation up to coefficient seven resulted in an accuracy of $88.12\%$, falling short of the minimum requirement of $90\%$. Therefore, a comprehensive exploration of the approach identified in the previous test case becomes necessary. It is important to highlight that Chebyshev series computation can be transformed into a simple polynomial evaluation. Given the low multiplicative depth, it becomes evident that converting the Chebyshev computation to polynomial evaluation is crucial for a thorough investigation of the limitations of the aforementioned technique.

An example of coefficients generated for evaluating an unscaled polynomial of degree 16 are as follows: **{0.5, 0.19, 0.0, -0.004, 0.0, 4.83e-05, 0.0, -2.97e-07, 0.0, 1.02e-09, 0.0, -1.94e-12, 0.0, 1.94e-15, 0.0, -7.89e-19, 0.0}**. Notably, the values of these coefficients decrease progressively. This diminishing trend arises because the scaling factor, previously managed by the ciphertext, is now concentrated at the coefficient level. We compute the evaluation up to $x^7$ in a very straightforward manner, as shown below.

\begin{align}
     \texttt{SUM}_7 = (c_7\times x^4)\times x^3 +(c_5\times x^3)\times x^2 +(c_3\times x^2)\times x + c_1\times x+ c_0 
\end{align}

<img src="https://hackmd.io/_uploads/HJTLXC636.png"  alt="drawing" width="28" /> For the subsequent evaluation, it is apparent that coefficients 9, 11, 13, and 15 are exceedingly small, nearly reaching the limits of the scaling factor. It will result in computational errors if not handled cautiously. Hence, we recommend breaking down the computation of higher coefficients into smaller chunks, as outlined below:

\begin{align}
    y_1 && = && (10^{-3}\times x )\times x && \text{// $10^{-3}\times x^2$ at depth 2}\\
    y_2 && = && (c_9\times 10^{6} \times x )\times x && \text{// $c_9 \times 10^{6}\times x^2$ at depth 2}\\
    c_9\times x^9 && = && (y_1\times y_1) \times (y_2\times x^3) &&
\end{align}

<img src="https://hackmd.io/_uploads/HJTLXC636.png"  alt="drawing" width="28" /> Applying this approach, we can extend the evaluation to $x^{13}$ as depicted in the equation below. This limitation arises because the coefficient computation needs to be divided into at least two parts. The maximum degree achievable with two constant multiplications in this manner is $x^{14}$. Thus, we have reached the limit of this technique. 

\begin{align}
    y_1 && = && (10^{-6}\times x )\times x^2 && \text{//$10^{-6}\times x^3$ at depth 2}\\
    y_2 && = && (c_{13}\times 10^{12} \times x )\times x^2 && \text{// $c_{13} \times 10^{12}\times x^3$ at depth 2}\\
    c_{13}\times x^{13} && = && (y_1\times y_1) \times (y_2\times x^4) &&
\end{align}

The computation up to $x^{13}$ led to an enhancement in accuracy from $88.12\%$ to $96.6\%$ <img src="https://hackmd.io/_uploads/ryckn_n3a.png" alt="drawing" width="45" />, which is a significant improvement in the attainable accuracy. Exploring the effectiveness of this technique in approximating functions other than logistic functions at different depths would be an interesting avenue for future exploration.

___
## <img src="https://hackmd.io/_uploads/BknuCw_pT.png"  alt="drawing" width="30"/>  Ready, set, LOG! 

Dive into the private Logistic function implementations live at: https://github.com/fairmath/components/

___

[KSK+18]: http://eprint.iacr.org/2018/254
[KSW+18]: http://eprint.iacr.org/2018/074
[XWBB16]: http://arxiv.org/abs/1611.01170
[AHPW16]: https://doi.org/10.1145/2857705.2857731
[LLNK22]: https://doi.org/10.1109/TDSC.2021.3105111
[image]: ![](https://hackmd.io/_uploads/By08S_OTp.png)


<p style="text-align:center;font-family:georgia,garamond;">  Made with ♥️ </p>
