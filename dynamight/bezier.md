
N=5
---
$$ B(t) = \sum_{i=0}^n {n \choose i} (1-t)^{n-i} t^i P_i $$
$$ b(n,i) = \frac{n!}{i! (n-i)!} $$

$$ B(t) = (1-t)^5 P_0 + 5 (1-t)^4 t P_1 + 10 (1-t)^3 t^2 P_2 + 10 (1-t)^2 t^3 P_3 + 5  (1-t)   t^4 P_4 + t^5 P_5 $$

For 0 < i < n

$$ \frac{d(1-t)}{dt} = -1 $$
$$ (fg)' = f'g + g'f $$
$$ B(i) = (1-t)^{n-i} t^i $$
$$ B'(i) = -(n-i)(1-t)^{n-i-1} t^i + i(1-t)^{n-i} t^{i-1} $$
$$ B'(i) = t^{i-1} t^{n-i-1} ( -(n-i)t + i(1-t)) $$
$$ B'(i) = (i-nt) (t^{i-1} t^{n-i-1}) $$
$$ C'(i) = \frac{n!}{i! (n-i)!} $$

so:

$$ \frac{dB}{dt} = -5(1-t)^4 P_0 $$
$$ + 5  (-4(1-t)^{5-2} t^1 + 1(1-t)^{4} t^0) = -20 (1-t)^3 t   +  5(1-t)^4     $$ 
$$ + 10 (-3(1-t)^{5-3} t^2 + 2(1-t)^{3} t^1) = -30 (1-t)^2 t^2 + 20(1-t)^3 t   $$ 
$$ + 10 (-2(1-t)^{5-4} t^3 + 3(1-t)^{2} t^2) = -20 (1-t)^1 t^3 + 30(1-t)^2 t^2 $$ 
$$ + 5  (-1(1-t)^{5-5} t^4 + 4(1-t)^{1} t^3) = -5          t^4 + 20(1-t)^1 t^3 $$ 
$$ + 5 t^4 P_5 $$ 

Let:

$$ B(i=0) = (1-t)^n P_0 $$
$$ B(i=n) = t^n P_n $$
$$ B'(i=0) = -n (1-t)^{n-1} P_0 $$
$$ B'(i=n) = n t^{n-1} P_n $$

$$ B'(t,n) = -n (1-t)^{n-1} P_0 + n t^{n-1} P_n + \sum_{i=1}^{n-1} \frac{n!}{i! (n-i)!} (-(n-i)(1-t)^{n-i-1} t^i + i(1-t)^{n-i} t^{i-1}) $$

Definition
----------
Per the explicit definition,

$$ B(t) = \sum_{i=0}^n {n \choose i} (1-t)^{n-i} t^i P_i $$

https://en.wikipedia.org/wiki/B%C3%A9zier_curve

where

$$ {n \choose i} = \frac{n!}{i! (n-i)!} \text{ for } 0 \le i \le n $$

https://en.wikipedia.org/wiki/Binomial_coefficient

N=5
---
$$ B(t) = \sum_{i=0}^n {n \choose i} (1-t)^{n-i} t^i P_i $$

$$ b(n,i) = n! / (i! (n-i)!) $$

$$ B(t) = $$
$$ + {5 \choose 0} (1-t)^{5}   t^0 P_0 = \frac{5!}{0! 5!} ... = 1  (1-t)^5     P_0 $$
$$ + {5 \choose 1} (1-t)^{5-1} t^1 P_1 = \frac{5!}{1! 4!} ... = 5  (1-t)^4 t   P_1 $$
$$ + {5 \choose 2} (1-t)^{5-2} t^2 P_2 = \frac{5!}{2! 3!} ... = 10 (1-t)^3 t^2 P_2 $$
$$ + {5 \choose 3} (1-t)^{5-3} t^3 P_3 = \frac{5!}{3! 2!} ... = 10 (1-t)^2 t^3 P_3 $$
$$ + {5 \choose 4} (1-t)^{5-4} t^4 P_4 = \frac{5!}{4! 1!} ... = 5  (1-t)   t^4 P_4 $$
$$ + {5 \choose 5} (1-t)^{5-5} t^5 P_5 = \frac{5!}{5! 0!} ... = 1          t^5 P_5 $$
 
$$ B(t) = (1-t)^5 P_0 + 5 (1-t)^4 t P_1 + 10 (1-t)^3 t^2 P_2 + 10 (1-t)^2 t^3 P_3 + 5  (1-t)   t^4 P_4 + t^5 P_5 $$
