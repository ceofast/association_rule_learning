# ASSOCIATION RULE LEARNING (BİRLİKTELİK KURALI ÖĞRENİMİ)

# The association rules method is to present these correlations in the best way through
# rules, if there are significant correlations between the items that occur simultaneously
# and frequently, and if there are significant correlations. In other words, it is a
# rule-based machine learning technique used to find patterns in data.

# There is a very big problem in social media channels, on the basis of e-commerce resources
# on the internet. There are hundreds of thousands of content on such sites and they are
# stored in their databases. We cannot upload these hundreds of thousands of content to the user.
# We should use content filtering methods. When we watch or like a video, we enter a certain flow.
# It takes the best extract of that flow and personalizes us. Basically, our purpose in these systems
# is to filter the contents.
# Source;
#       https://www.datasciencearth.com/birliktelik-kurallari-algoritmalari/
#       https://www.veribilimiokulu.com/category/makine-ogrenmesi/

#   - Apriori Algorithm -

# It is a basket analysis method and is used to reveal product associations.

# Support(X, Y) = Freq(X, Y) / N
# There are 3 very simple formulas. The 1st is the Support value. It expresses the probability of
# X and Y occurring together. It is the frequency of X and Y appearing together divided by N.

# Confidence(X, Y) = Freq(X, Y) / Freq(X)
# It expresses the probability of purchasing product Y when product X is purchased.
# The frequency at which X and Y appear together divided by the frequency at which X appears.

# Lift = Support(X, Y) / (Support(x) * Support (Y))
# When X is purchased, the probability of buying Y increases by a multiple of lift.
# The probability of X and Y appearing together is the product of the probabilities
# of X and Y appearing separately.
# It states an expression such as how many times the probability of buying another product
# increases when we buy a product.

# Our aim is to suggest products to users in the product purchasing process by
# applying association analysis to the online retail II dataset.

# 1. Data Pre-processing

# !pip install mlxtend

import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
# It ensures that the output is on one line.
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

df_ = pd.read_excel("/Users/cenancanbikmaz/PycharmProjects/DSMLBC-7/HAFTA_3/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()

df.info()

# Column       Non-Null Count   Dtype
# ---  ------       --------------   -----
#  0   Invoice      541910 non-null  object
#  1   StockCode    541910 non-null  object
#  2   Description  540456 non-null  object
#  3   Quantity     541910 non-null  int64
#  4   InvoiceDate  541910 non-null  datetime64[ns]
#  5   Price        541910 non-null  float64
#  6   Customer ID  406830 non-null  float64
#  7   Country      541910 non-null  object

df.head()

#  Invoice StockCode                          Description  Quantity         InvoiceDate  Price  Customer ID         Country
# 0  536365    85123A   WHITE HANGING HEART T-LIGHT HOLDER         6 2010-12-01 08:26:00   2.55      17850.0  United Kingdom
# 1  536365     71053                  WHITE METAL LANTERN         6 2010-12-01 08:26:00   3.39      17850.0  United Kingdom
# 2  536365    84406B       CREAM CUPID HEARTS COAT HANGER         8 2010-12-01 08:26:00   2.75      17850.0  United Kingdom
# 3  536365    84029G  KNITTED UNION FLAG HOT WATER BOTTLE         6 2010-12-01 08:26:00   3.39      17850.0  United Kingdom
# 4  536365    84029E       RED WOOLLY HOTTIE WHITE HEART.         6 2010-12-01 08:26:00   3.39      17850.0  United Kingdom

# We use this function to determine the threshold values of the data.
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

# This function also replaces the determined outlier threshold values with outliers.
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# In this function, we extract the values containing 'C' from the data. "C" means returned items.
# To calculate Total Price, the variables Quantity and Price must be greater than zero.
# We close the function by calling the Outlier and Threshold functions.
def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe

df = retail_data_prep(df)

# 2. Preparing the ARL Data Structure (Invoice-Product Matrix)

# The view we want the data to come from.

# Description   NINE DRAWER OFFICE TIDY   SET 2 TEA TOWELS I LOVE LONDON    SPACEBOY BABY GIFT SET
# Invoice
# 536370                              0                                 1                       0
# 536852                              1                                 0                       1
# 536974                              0                                 0                       0
# 537065                              1                                 0                       0
# 537463                              0                                 0                       1

# First we will try to put the invoices on the lines because they will be our cart.
df_gr = df[df['Country'] == 'Germany']

df_gr.head()

#  Invoice StockCode                          Description  Quantity         InvoiceDate  Price  Customer ID  Country
# 1109  536527     22809              SET OF 6 T-LIGHTS SANTA       6.0 2010-12-01 13:04:00   2.95      12662.0  Germany
# 1110  536527     84347  ROTATING SILVER ANGELS T-LIGHT HLDR       6.0 2010-12-01 13:04:00   2.55      12662.0  Germany
# 1111  536527     84945   MULTI COLOUR SILVER T-LIGHT HOLDER      12.0 2010-12-01 13:04:00   0.85      12662.0  Germany
# 1112  536527     22242        5 HOOK HANGER MAGIC TOADSTOOL      12.0 2010-12-01 13:04:00   1.65      12662.0  Germany
# 1113  536527     22244           3 HOOK HANGER MAGIC GARDEN      12.0 2010-12-01 13:04:00   1.95      12662.0  Germany

# According to Invoice and Description, we got groupby and counted Quantities.
# We said how many of this product are on this invoice.
df_gr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).head(20)

# Invoice Description
# 536527  3 HOOK HANGER MAGIC GARDEN               12.0
#         5 HOOK HANGER MAGIC TOADSTOOL            12.0
#         5 HOOK HANGER RED MAGIC TOADSTOOL        12.0
#         ASSORTED COLOUR LIZARD SUCTION HOOK      24.0
#         CHILDREN'S CIRCUS PARADE MUG             12.0
#         HOMEMADE JAM SCENTED CANDLES             12.0
#         HOT WATER BOTTLE BABUSHKA                 4.0
#         JUMBO BAG OWLS                           10.0
#         JUMBO BAG WOODLAND ANIMALS               10.0
#         MULTI COLOUR SILVER T-LIGHT HOLDER       12.0
#         PACK 3 FIRE ENGINE/CAR PATCHES           12.0
#         PICTURE DOMINOES                         12.0
#         POSTAGE                                   1.0
#         ROTATING SILVER ANGELS T-LIGHT HLDR       6.0
#         SET OF 6 T-LIGHTS SANTA                   6.0
# 536840  6 RIBBONS RUSTIC CHARM                   12.0
#         60 CAKE CASES VINTAGE CHRISTMAS          24.0
#         60 TEATIME FAIRY CAKE CASES              24.0
#         CAKE STAND WHITE TWO TIER LACE            2.0
#         JAM JAR WITH GREEN LID                   12.0

# We use unstack to avoid multiplexing and we use iloc to show the first 5 observations.
# If a product is on an invoice, we did it this way to show how many information came from that product.
# If a product is not in the cart(invoice), NA will come.
df_gr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().iloc[0:5, 0:5]

# Description  50'S CHRISTMAS GIFT BAG LARGE  DOLLY GIRL BEAKER  I LOVE LONDON MINI BACKPACK  RED SPOT GIFT BAG LARGE  SET 2 TEA TOWELS I LOVE LONDON
# Invoice
# 536527                                 NaN                NaN                          NaN                      NaN                              NaN
# 536840                                 NaN                NaN                          NaN                      NaN                              NaN
# 536861                                 NaN                NaN                          NaN                      NaN                              NaN
# 536967                                 NaN                NaN                          NaN                      NaN                              NaN
# 536983                                 NaN                NaN                          NaN                      NaN                              NaN

# We need one hot encoded version. We want to write 0 where it says NA.
df_gr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().fillna(0).iloc[0:5, 0:5]

# Description  50'S CHRISTMAS GIFT BAG LARGE  DOLLY GIRL BEAKER  I LOVE LONDON MINI BACKPACK  RED SPOT GIFT BAG LARGE  SET 2 TEA TOWELS I LOVE LONDON
# Invoice
# 536527                                 0.0                0.0                          0.0                      0.0                              0.0
# 536840                                 0.0                0.0                          0.0                      0.0                              0.0
# 536861                                 0.0                0.0                          0.0                      0.0                              0.0
# 536967                                 0.0                0.0                          0.0                      0.0                              0.0
# 536983                                 0.0                0.0                          0.0                      0.0                              0.0

# Now we're going to do something a little different than the last one we did.
# Here, we will write 1 if the products in the invoices are greater than 0 in quantity.
# We will write 0 if it is less than 0 or 0. We were operating on rows or columns with apply.
# Here we will go through all the cells by applying the applymap and perform the operation.
df_gr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0).iloc[0:5, 0:5]

# Description  50'S CHRISTMAS GIFT BAG LARGE  DOLLY GIRL BEAKER  I LOVE LONDON MINI BACKPACK  RED SPOT GIFT BAG LARGE  SET 2 TEA TOWELS I LOVE LONDON
# Invoice
# 536527                                   0                  0                            0                        0                                0
# 536840                                   0                  0                            0                        0                                0
# 536861                                   0                  0                            0                        0                                0
# 536967                                   0                  0                            0                        0                                0
# 536983                                   0                  0                            0                        0                                0

# We create a function called create_invoice_product_df. If we want to search according
# to the id variable and get results, it will do the same as above according to the stockcode.
# If we entered the id as False, it will perform the above operation according to Desceiptions.
def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)

gr_inv_pro_df = create_invoice_product_df(df_gr)
gr_inv_pro_df.head(20)

gr_inv_pro_df = create_invoice_product_df(df_gr, id=True)
gr_inv_pro_df.head()

def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)

check_id(df_gr, 16016)

# ['LARGE CHINESE STYLE SCISSOR']


# 3. Possibilities of All Possible Product Combinations

# Support(X, Y) = Freq(X, Y) / N
# There are 3 very simple formulas. The 1st is the Support value. It expresses the probability of
# X and Y occurring together. It is the frequency of X and Y appearing together divided by N.

# Confidence(X, Y) = Freq(X, Y) / Freq(X)
# It expresses the probability of purchasing product Y when product X is purchased.
# The frequency at which X and Y appear together divided by the frequency at which X appears.

# Lift = Support(X, Y) / (Support(x) * Support (Y))
# When X is purchased, the probability of buying Y increases by a multiple of lift.
# The probability of X and Y appearing together is the product of the probabilities
# of X and Y appearing separately.
# It states an expression such as how many times the probability of buying another product
# increases when we buy a product.

#If there is a possibility of appearing together in this function, whatever value we enter
# in min_support will not take into account the values below those values.
frequent_itemsets = apriori(gr_inv_pro_df, min_support=0.01, use_colnames=True)

frequent_itemsets.sort_values("support", ascending=False).head()

#  support       itemsets
# 538   0.818381         (POST)
# 189   0.245077        (22326)
# 1864  0.225383  (POST, 22326)
# 191   0.157549        (22328)
# 1931  0.150985  (22328, POST)

check_id(df_gr, 22328)
#['ROUND SNACK BOXES SET OF 4 FRUITS ']

# By inserting the support values we found with Apriori into the association_rules function,
# we find some other statistical data such as cofidance and lift.
rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)

rules.sort_values("support", ascending=False).head()
# According to this table, the probability of POST product and product numbered 22326
# appearing together is 0.225383. The probability of being bought together is 0.275401.
# The increase in the probability of buying these two products together is 1.123735.

# antecedents consequents  antecedent support  consequent support   support  confidence      lift  leverage  conviction
# 2650      (POST)     (22326)            0.818381            0.245077  0.225383    0.275401  1.123735  0.024817    1.041850
# 2651     (22326)      (POST)            0.245077            0.818381  0.225383    0.919643  1.123735  0.024817    2.260151
# 2784     (22328)      (POST)            0.157549            0.818381  0.150985    0.958333  1.171012  0.022049    4.358862
# 2785      (POST)     (22328)            0.818381            0.157549  0.150985    0.184492  1.171012  0.022049    1.033038
# 2414     (22328)     (22326)            0.157549            0.245077  0.131291    0.833333  3.400298  0.092679    4.529540

rules.sort_values("lift", ascending=False).head(5)
#                 antecedents            consequents  antecedent support  consequent support   support  confidence  lift  leverage  conviction
# 24744         (21987, 21988)         (21989, 21086)            0.010941            0.010941  0.010941         1.0  91.4  0.010821         inf
# 39026  (21987, 21988, 21094)         (21989, 21086)            0.010941            0.010941  0.010941         1.0  91.4  0.010821         inf
# 39043         (21989, 21086)  (21987, 21988, 21094)            0.010941            0.010941  0.010941         1.0  91.4  0.010821         inf
# 24749         (21989, 21086)         (21987, 21988)            0.010941            0.010941  0.010941         1.0  91.4  0.010821         inf
# 39036         (21987, 21989)  (21086, 21988, 21094)            0.010941            0.010941  0.010941         1.0  91.4  0.010821         inf

# This content belongs to "Data Science School and Miuul". It cannot be used without permission.





