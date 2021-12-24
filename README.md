ASSOCIATION RULE LEARNING

The association rules method is to present these correlations in the best way through
rules, if there are significant correlations between the items that occur simultaneously
and frequently, and if there are significant correlations. In other words, it is a
rule-based machine learning technique used to find patterns in data.

There is a very big problem in social media channels, on the basis of e-commerce resources
on the internet. There are hundreds of thousands of content on such sites and they are
stored in their databases. We cannot upload these hundreds of thousands of content to the user.
We should use content filtering methods. When we watch or like a video, we enter a certain flow.
It takes the best extract of that flow and personalizes us. Basically, our purpose in these systems
is to filter the contents.
Source;
       https://www.datasciencearth.com/birliktelik-kurallari-algoritmalari/
       https://www.veribilimiokulu.com/category/makine-ogrenmesi/

   - Apriori Algorithm -

It is a basket analysis method and is used to reveal product associations.

Support(X, Y) = Freq(X, Y) / N
There are 3 very simple formulas. The 1st is the Support value. It expresses the probability of
X and Y occurring together. It is the frequency of X and Y appearing together divided by N.

Confidence(X, Y) = Freq(X, Y) / Freq(X)
It expresses the probability of purchasing product Y when product X is purchased.
The frequency at which X and Y appear together divided by the frequency at which X appears.

Lift = Support(X, Y) / (Support(x) * Support (Y))
When X is purchased, the probability of buying Y increases by a multiple of lift.
The probability of X and Y appearing together is the product of the probabilities
of X and Y appearing separately.
It states an expression such as how many times the probability of buying another product
increases when we buy a product.

Our aim is to suggest products to users in the product purchasing process by
applying association analysis to the online retail II dataset.

# This content belongs to "Veri Bilimi Okulu and Miuul". It cannot be used without permission.
