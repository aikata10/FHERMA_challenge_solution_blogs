# SIGN Evaluation Challenge 

*The article details the solution provided by the winner of the SIGN  Evaluation Challenge.*

#### Author: [Aikata (Ph.D. student, TU Graz)](https://www.iaik.tugraz.at/person/aikata-aikata/)

___


## What is SIGN function <img src="https://hackmd.io/_uploads/SJBg0P3hp.png" alt="drawing" width="20"/>

<!--- ![Screenshot from 2024-02-29 11-01-11](https://hackmd.io/_uploads/SkzZ8Ca36.png)
--->



<p style="text-align: center;"><img src="https://hackmd.io/_uploads/SktwuAp2p.png" width="650" /> <br> <small><small> Inspired from <a href="https://www.imaginary.org/gallery/maths-dance-moves">Imaginary</a></small></small>
</p>




The sign function, also known as signum, determines the sign of a value. Its significance extends to various applications in machine learning (ML), serving as a foundational element for non-linear activation functions like the Rectified Linear Unit (ReLu) or Max Pooling. This is attributed to the ability of the sign function to facilitate comparison or max operations as follows:
\begin{align}
    \text{comp}(a,b) &= \frac{ \text{sign}(a-b)+1}{2}\\
    \text{max}(a,b) &= \frac{ (a+b)+(a-b) \text{sign}(a-b)}{2}
\end{align}



Fully Homomorphic Encryption (FHE) schemes like FHEW or TFHE can precisely compute the sign function using bootstrapping-based function evaluation, resembling a look-up table. However, in approximate or integer arithmetic schemes like CKKS, BGV, BFV, etc., the absence of table look-up capabilities makes evaluating non-polynomial discontinuous functions challenging. Consequently, these schemes resort to function approximation for sign evaluation. For example, Figure 1(a) demonstrates the effective use of Tanh with a modified domain (e.g., $x, 2\times x, \cdots, 128\times x$) to approximate the sign function. This helps bridge the discontinuity of the sign function using a continuous function. Note that Tanh is not a polynomial function. Hence, a polynomial approximation of Tanh, such as by using the Taylor series, is applied to approximate the sign function.

<p float="left">
  <img src="https://hackmd.io/_uploads/H1mcK48h6.png" width="350" />
  <img src="https://hackmd.io/_uploads/Bkj5FNI3T.png" width="350" /> 
</p>

<p style="text-align: center;"><b>Figure 1:</b> Function approximations for the sign function using a) Tanh and b) Chebyshev.</p>

In recent years, significant progress has been made in the field of FHE, leading researchers to explore methods for effectively approximating the sign function. One prevalent approach involves leveraging the Chebyshev series, known for its universal applicability in approximating a broad spectrum of functions. Notably, OpenFHE has already incorporated an implementation of the Chebyshev series. This implementation evaluates the Homomorphic modular reduction during the Bootstrapping procedure.


## <img src="https://hackmd.io/_uploads/SkcCXdh2p.png" alt="drawing" width="60"/> &nbsp; The FHERMA Challenge

The challenge has test cases with values $\in [-1,1]$ and allowed multiplicative depth of ten. 

An optimization of the sign approximation technique was introduced in [[LLNK22]]. In this research, the authors advocate for a sign approximation approach based on polynomial composition, with Chebyshev as the basis polynomial. They suggest and validate the effectiveness of dividing the computation depth, denoted as $d$, into smaller segments (e.g., $d_1,d_2$ where $d_1+d_2=d$). The proposed methodology initially evaluates a function that approximates the Chebyshev series for depth $d_1$. The obtained result is then utilized as an input for the Chebyshev series of depth $d_2$.

This approach could not be utilized to achieve high accuracy. Therefore, next we will delve into what approach worked for achieving the FHERMA challenge result. 

## <img src="https://hackmd.io/_uploads/BJUDuu336.png" alt="drawing" width="50"/>  &nbsp; The Solution
Initially, employing a straightforward approximation of the sign function through the Chebyshev series evaluation within the OpenFHE library yields an accuracy of $99.96\%$. It is important to note that the Chebyshev series can only be evaluated up to the coefficient $1006$ when approximating $\text{Tanh}(x\times\texttt{RAND_MAX})$ using OpenFHE. The resulting approximation is illustrated in Figure 1(b). To enhance the computational capabilities of OpenFHE, we conducted an analysis to determine the feasibility of evaluating additional Chebyshev series. Since the sign function is an odd function, the even degree coefficients do not contribute to its approximation. Hence, there is no need to evaluate the even terms of the Chebyshev series.

Building on this observation, we explored whether it is possible to evaluate $c \times\textbf{T}_{1009}$ while adhering to the multiplicative depth limitation of 10, where $c$ is the coefficient resulting from the function approximation. We utilize the properties of the Chebyshev series and write down recursive relations as follows:
\begin{align}
    \text{c}\times\textbf{T}_{1009} && \rightarrow && \text{2c}\times\textbf{T}_{512}\times\textbf{T}_{497}-\text{c}\times\textbf{T}_{15} && \rightarrow &&  \textbf{T}_{512}\times(\text{2c}\times\textbf{T}_{497})-\text{c}\times\textbf{T}_{15} \\
    \text{2c}\times\textbf{T}_{497} && \rightarrow &&  \text{4c}\times\textbf{T}_{256}\times\textbf{T}_{241}-\text{2c}\times\textbf{T}_{15} && \rightarrow  && \textbf{T}_{256}\times(\text{4c}\times\textbf{T}_{241})-\text{2c}\times\textbf{T}_{15} \\
    \text{4c}\times\textbf{T}_{241} && \rightarrow  && \text{8c}\times\textbf{T}_{128}\times\textbf{T}_{113}-\text{4c}\times\textbf{T}_{15} && \rightarrow &&  \textbf{T}_{128}\times(\text{8c}\times\textbf{T}_{113})-\text{4c}\times\textbf{T}_{15}\\
    \text{8c}\times\textbf{T}_{113} && \rightarrow &&  \text{16c}\times\textbf{T}_{64}\times\textbf{T}_{49}-\text{8c}\times\textbf{T}_{15} && \rightarrow &&  \textbf{T}_{64}\times(\text{16c}\times\textbf{T}_{49})-\text{8c}\times\textbf{T}_{15}\\
    \text{16c}\times\textbf{T}_{49} && \rightarrow &&  \text{32c}\times\textbf{T}_{32}\times\textbf{T}_{17}-\text{16c}\times\textbf{T}_{15} && \rightarrow &&  \textbf{T}_{32}\times(\text{32c}\times\textbf{T}_{17})-\text{16c}\times\textbf{T}_{15}\\
    \text{32c}\times\textbf{T}_{17} && \rightarrow &&  \text{64c}\times\textbf{T}_{16}\times\textbf{T}_{1}-\text{32c}\times\textbf{T}_{15} && \rightarrow &&  \textbf{T}_{16}\times(\text{64c}\times\textbf{T}_{1})-\text{32c}\times\textbf{T}_{15}
\end{align}

<img src="https://hackmd.io/_uploads/HJTLXC636.png"  alt="drawing" width="28" /> Upon the initial breakdown of $\textbf{T}_{1009}$, we observed that the series terms to be multiplied $(\textbf{T}_{512},$ $\textbf{T}_{497}$) were at the same depth of nine. Unfortunately, this configuration made it impossible to perform a third multiplication with the coefficient $\text{c}$. Therefore, we recursively broke down the expression until we achieved terms ($\textbf{T}_{16}$, $\textbf{T}_{1}$) at different depths (4,1). At this stage, we can multiply $\text{c}$ with $\textbf{T}_{1}$, bringing it to a depth of one. Subsequently, we can compute the recursive breakdown upwards to obtain $c \times \textbf{T}_{1009}$. This strategy enables us to evaluate more Chebyshev series coefficients effectively.

<img src="https://hackmd.io/_uploads/HJTLXC636.png"  alt="drawing" width="28"/> The same methodology is applied to evaluate all the remaining Chebyshev series coefficients - $\textbf{T}_{1011}$, $\textbf{T}_{1013}$, $\textbf{T}_{1015}$, $\textbf{T}_{1017}$, $\textbf{T}_{1019}$, $\textbf{T}_{1021}$, and $\textbf{T}_{1023}$. It is crucial to note that the recursion must be further broken down to evaluate higher degree coefficients. This additional evaluation increases accuracy from $99.96\%$ to $99.97\%$ <img src="https://hackmd.io/_uploads/BJx3CY_23p.png" alt="drawing" width="35"/>. The approximation for this is shown in Figure 1 (b). With this, we conclude the FHERMA challenge solution for the SIGN problem. 

___
## <img src="https://hackmd.io/_uploads/BknuCw_pT.png"  alt="drawing" width="30"/>  Follow the SIGNs!

Dive into the private Signum function implementation live at: https://github.com/Fherma-challenges/signum
___


[LLNK22]: https://doi.org/10.1109/TDSC.2021.3105111


<p style="text-align:center;font-family:georgia,garamond;">  Made with ♥️ </p>


