import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from lmfit import minimize, Parameters, fit_report
import statistics as stats
from sklearn.linear_model import LinearRegression
from scipy.stats import ttest_1samp, ttest_ind, t
publisher_data=pd.read_csv("publishers-corgis.csv")
print(publisher_data.isna().sum())

pd.set_option('display.max_columns', None)
with pd.option_context('display.max_rows', 50):
    print(publisher_data.head(50))
publisher_data1=publisher_data.dropna()
print(publisher_data)
num_zeros = (publisher_data==0).sum()

print(num_zeros)


publisher_data1 = publisher_data[publisher_data != 0].dropna()



print(publisher_data1)

# Calculate the daily average revenue for each genre
daily_avg_revenue = publisher_data.groupby('genre')['daily average.amazon revenue'].mean().sort_values(ascending=False)
print("daily average.amazon revenue : ",daily_avg_revenue)
# Calculate the average rating for each genre
avg_rating = publisher_data1.groupby('genre')['statistics.average rating'].mean().sort_values(ascending=False)
print("avg_rating  : ",avg_rating )
# Calculate the total reviews for each genre
total_reviews = publisher_data1.groupby('genre')['statistics.total reviews'].sum().sort_values(ascending=False)
print("total_reviews : ",total_reviews )
# Calculate the daily average gross sales for each genre
daily_avg_gross_sales =publisher_data.groupby('genre')['daily average.gross sales'].mean().sort_values(ascending=False)
print("daily_avg_gross_sales  : ",daily_avg_gross_sales  )
# Calculate the most sold units daily for each genre
most_sold_units_daily = publisher_data.groupby('genre')['daily average.units sold'].max().sort_values(ascending=False)
print("most_sold_units_daily: ",most_sold_units_daily  )
#Calculating sales prices
avg_statistics_sale_price  = publisher_data.groupby('genre')["statistics.sale price"].mean().sort_values(ascending=False)
print("daily avg_statistics.sale price : ",avg_statistics_sale_price)

print()
print()
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print()
print()



import numpy as np

# Calculate the daily average revenue for each genre
daily_avg_revenue = publisher_data.groupby('genre').agg({'daily average.amazon revenue': [np.mean, np.std, np.min, np.max]})
print("daily average.amazon revenue : ", daily_avg_revenue)
# Calculate the average rating for each genre
avg_rating = publisher_data1.groupby('genre').agg({'statistics.average rating': [np.median, np.std, np.min, np.max]})
print("avg_rating  : ", avg_rating)
# Calculate the total reviews for each genre
total_reviews = publisher_data1.groupby('genre').agg({'statistics.total reviews': [np.mean, np.std, np.min, np.max]})
print("total_reviews : ", total_reviews)
# Calculate the daily average gross sales for each genre
daily_avg_gross_sales = publisher_data.groupby('genre').agg({'daily average.gross sales': [np.mean, np.std, np.min, np.max]})
print("daily_avg_gross_sales  : ", daily_avg_gross_sales)
# Calculate the most sold units daily for each genre
most_sold_units_daily = publisher_data.groupby('genre').agg({'daily average.units sold': [np.mean, np.std, np.min, np.max]})
print("most_sold_units_daily: ", most_sold_units_daily)
# Calculate the average sale price for each genre
avg_statistics_sale_price = publisher_data.groupby('genre').agg({'statistics.sale price': [np.mean, np.std, np.min, np.max]})
print("avg_statistics_sale_price: ", avg_statistics_sale_price)



import pandas as pd

# Load the data into a DataFrame

# Calculate the daily average revenue for each genre
daily_avg_revenue = publisher_data.groupby('genre').agg({'daily average.amazon revenue': 'mean'})

# Find the genre with the highest daily average revenue
highest_daily_avg_revenue_genre = daily_avg_revenue['daily average.amazon revenue'].idxmax()
print("Genre with highest daily average revenue: ", highest_daily_avg_revenue_genre)

# Calculate the average rating for each genre
avg_rating = publisher_data1.groupby('genre').agg({'statistics.average rating': 'median'})

# Find the genre with the highest average rating
highest_avg_rating_genre = avg_rating['statistics.average rating'].idxmax()
print("Genre with highest average rating: ", highest_avg_rating_genre)

# Calculate the total reviews for each genre
total_reviews = publisher_data1.groupby('genre').agg({'statistics.total reviews': 'mean'})

# Find the genre with the highest total reviews
highest_total_reviews_genre = total_reviews['statistics.total reviews'].idxmax()
print("Genre with highest total reviews: ", highest_total_reviews_genre)

# Calculate the daily average gross sales for each genre
daily_avg_gross_sales = publisher_data.groupby('genre').agg({'daily average.gross sales': 'mean'})

# Find the genre with the highest daily average gross sales
highest_daily_avg_gross_sales_genre = daily_avg_gross_sales['daily average.gross sales'].idxmax()
print("Genre with highest daily average gross sales: ", highest_daily_avg_gross_sales_genre)

# Calculate the most sold units daily for each genre
most_sold_units_daily = publisher_data.groupby('genre').agg({'daily average.units sold': 'mean'})

# Find the genre with the most sold units daily
most_sold_units_daily_genre = most_sold_units_daily['daily average.units sold'].idxmax()
print("Genre with the most sold units daily: ", most_sold_units_daily_genre)
avg_statistics_sale_price = publisher_data1.groupby('genre').agg({'statistics.sale price': np.mean})
highest_avg_statistics_sale_price = avg_statistics_sale_price['statistics.sale price'].idxmax()
print("Genre with highest avg_statistics_sale_price: ",highest_avg_statistics_sale_price)




#part
sns.scatterplot(x="publisher.type", y="daily average.units sold", hue="publisher.type", data=publisher_data)
plt.title('The scatterplot of publisher type and daily average units sold')
plt.show()

# create dummy variables for publisher type
publisher_dummies = pd.get_dummies(publisher_data['publisher.type'])
# concatenate the dummy variables with the original data
publisher_data_dummies = pd.concat([publisher_data, publisher_dummies], axis=1)
# calculate the correlation between the dummy variables and daily average unit sold
corr = publisher_data_dummies.iloc[:,7:].corrwith(publisher_data_dummies['daily average.units sold'])
print(corr)


sns.scatterplot(data=publisher_data_dummies, x='publisher.type', y='daily average.units sold')
plt.title('The scatterplot of publisher type and daily average units sold')
plt.show()
print(publisher_data['publisher.type'].unique())

#part2
# count the number of books in each genre
#genre_counts_data=publisher_data["publisher.name"]
genre_counts = publisher_data["genre"].value_counts()

# print the most popular genre
print('The most popular genre is:', genre_counts.index[0])

print()
#8

# count the number of books in each genre
most_popular_genre = publisher_data["genre"].value_counts().index[0]
# print the most popular genre
print('The most popular genre is:',most_popular_genre)
#6

publisher_data['total_daily_seller_revenue'] =publisher_data['daily average.amazon revenue'] + publisher_data['daily average.author revenue']
publisher_data['seller_percentage'] = publisher_data['total_daily_seller_revenue'] / publisher_data['daily average.gross sales']
publisher_data['publisher_percentage'] = publisher_data1['daily average.publisher revenue'] / publisher_data['daily average.gross sales']
publisher_data['author_percentage'] = publisher_data['daily average.author revenue'] / publisher_data['daily average.gross sales']

# print the results
print(publisher_data[['sold by', 'publisher.name', 'seller_percentage', 'publisher_percentage', 'author_percentage']])


publisher_data['total_daily_seller_revenue'] =publisher_data['daily average.amazon revenue'] + publisher_data['daily average.author revenue']
mean_seller_revenue=np.mean(publisher_data['total_daily_seller_revenue'])
mean_publisher_revenue=np.mean(publisher_data1['daily average.publisher revenue'])
mean_author_revenue=np.mean(publisher_data['daily average.author revenue'] )
# print the results
print('Mean Seller Revenue:', mean_seller_revenue)
print('Mean Publisher Revenue:', mean_publisher_revenue)
print('Mean Author Revenue:', mean_author_revenue)
labels = ['Seller Revenue', 'Publisher Revenue', 'Author Revenue']
means = [mean_seller_revenue, mean_publisher_revenue, mean_author_revenue]
stds = [np.std(publisher_data['total_daily_seller_revenue']),
        np.std(publisher_data1['daily average.publisher revenue']),
        np.std(publisher_data['daily average.author revenue'])]

sns.barplot(x=labels, y=means, yerr=stds, width=0.1,capsize=2, error_kw={'elinewidth':1, 'capsize':3, 'capthick':1})
plt.title('Comparison of Revenue')
plt.ylabel('Mean Daily Revenue')
plt.show()


#---------------------------------------------------------------------------------------------------------


publisher_groups = publisher_data.groupby("publisher.type")
publisher_groups1 = publisher_data1.groupby("publisher.type")

# calculate summary statistics for each variable by publisher type
sales_stats = publisher_data.groupby("genre")["daily average.gross sales"].describe()[["mean", "std", "max", "min"]]
rating_stats = publisher_data1.groupby("genre")["statistics.average rating"].agg(['median', 'std', 'max', 'min'])
price_stats = publisher_data.groupby("genre")["statistics.sale price"].describe()[["mean", "std", "max", "min"]]
rank_stats = publisher_data.groupby("genre")["statistics.sales rank"].agg(['median', 'std', 'max', 'min'])
review_stats = publisher_data1.groupby("genre")["statistics.total reviews"].describe()[["mean", "std", "min"]]

# print the summary statistics
print("Sales Statistics by Publisher Type:\n", sales_stats)
print()
print("Rating Statistics by Publisher Type:\n", rating_stats)
print()
print("Price Statistics by Publisher Type:\n", price_stats)
print()
print("Rank Statistics by Publisher Type:\n", rank_stats)
print()
print("Review Statistics by Publisher Type:\n", review_stats)


sns.boxplot(data=publisher_data, x="publisher.type", y="daily average.gross sales", hue="publisher.type")
plt.ylim(-200,4000)
plt.title("Boxplot of Daily Average Gross Sales by Publisher Type")
plt.show()
sns.boxplot(data=publisher_data1, x="publisher.type", y="statistics.average rating")
plt.ylim(0, 20)
plt.title("Boxplot of Daily Average rating by Publisher Type")
plt.show()
sns.boxplot(data=publisher_data, x="publisher.type", y="statistics.sale price", hue="publisher.type")
plt.ylim(-1, 99)
plt.title("Boxplot of sale price by Publisher Type")
plt.show()
sns.boxplot(data=publisher_data, x="publisher.type", y="statistics.sales rank")
plt.title("Boxplot of Sales Rank by Publisher Type")
plt.show()
sns.boxplot(data=publisher_data1, x="publisher.type", y="statistics.total reviews", hue="publisher.type")
plt.ylim(-1, 2000)
plt.title("Boxplot of total reviews by Publisher Type")
plt.show()


print()





print()
print("end")

#-------
# group the data by publisher type
publisher_groups = publisher_data.groupby("publisher.type")
publisher_groups1 = publisher_data1.groupby("publisher.type")
# calculate mean and std for each variable by publisher type
sales_mean = publisher_groups["daily average.gross sales"].aggregate([np.mean, np.std])
rating_med = publisher_groups1["statistics.average rating"].aggregate(np.median)
price_mean = publisher_groups["statistics.sale price"].aggregate([np.mean, np.std])
rank_med = publisher_groups["statistics.sales rank"].aggregate(np.median)
review_mean = publisher_groups1["statistics.total reviews"].aggregate([np.mean, np.std])
# print mean and std for each variable by publisher type
print("Gross Sales Mean by Publisher Type:\n", sales_mean)
print()
print("\nRating Median by Publisher Type:\n", rating_med)
print()
print("\nSale Price Mean by Publisher Type:\n", price_mean)
print()
print("\nRank Median by Publisher Type:\n", rank_med)
print()
print("\nReview Mean by Publisher Type:\n", review_mean)
print()



print()
print("calculating for the mode of the publisher type")


publishertype_mode=stats.mode(publisher_data["publisher.type"])
print(publishertype_mode)



print("+++++++++++++++++++++++++++++++++++")




import seaborn as sns
import matplotlib.pyplot as plt

revenue_mean = publisher_data.groupby(['publisher.name', 'genre'])['daily average.publisher revenue'].mean().reset_index()
units_mean = publisher_data.groupby(['publisher.name', 'genre'])['daily average.units sold'].mean().reset_index()




def f(x, params):
    a1 = params['a1'].value
    a0 = params['a0'].value
    R = a1*x + a0
    return R

# Define the error function to be minimized
def E(params, x, R):
    f_vals = f(x, params)
    error = R - f_vals
    return error
# Set up the data
D = publisher_data["statistics.sales rank"]
R = publisher_data["statistics.average rating"]
params = Parameters()
params.add('a1', value=1)
params.add('a0', value=5, vary=False)
# Minimize the error function and report the fit results
result = minimize(E, params, args=(D, R))
# Calculate the fitted values and plot the data with the regression line
R_fit=f(D,result.params)
sns.regplot(x=D,y=R)
plt.title("the correlation between the rank and rating ")
plt.show()
# Calculate and print the correlation coefficient
correlation_coefficient = np.corrcoef(D, R)[0][1]
print("Correlation coefficient:", correlation_coefficient)


sns.regplot(x="statistics.sales rank", y="statistics.average rating", data=publisher_data, line_kws={"linewidth": 2})

# display correlation coefficient
corr_coef = publisher_data["statistics.sale price"].corr(publisher_data['statistics.average rating'])
print("Correlation coefficient:", corr_coef)

# show plot
plt.show()
sns.scatterplot(x="statistics.sale price", y="statistics.average rating", data=publisher_data)
plt.title("Relationship between Sale Price and Rating of Books")
plt.xlabel("Sale Price")
plt.ylabel("Rating")
plt.show()


from scipy.stats import pearsonr, ttest_ind
correlation, p_value = pearsonr(publisher_data["statistics.sale price"], publisher_data['statistics.average rating'])
print("Pearson correlation coefficient:", correlation)
print("p-value:", p_value)
# Perform a hypothesis test
alpha = 0.05
if p_value < alpha:
    print("Reject null hypothesis: There is a strong correlation between price and rating")
else:
    print("Fail to reject null hypothesis: There is not enough evidence to support a strong correlation between price and rating")




t_value,p_value=ttest_ind(publisher_data["statistics.sale price"],publisher_data['statistics.average rating'],equal_var= False)

print("the p-value:",p_value)
