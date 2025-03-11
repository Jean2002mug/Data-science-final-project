# Data Analysis on Book Sales and Publisher Impact

## Introduction
The book industry is a dynamic market influenced by various factors, including genre, publisher type, and pricing strategies. This analysis examines book sales data to identify trends in genre popularity, publisher influence, and revenue distribution. By leveraging data science techniques, we aim to uncover insights that can inform decision-making for publishers, authors, and retailers.

For the full analysis, you can read the complete document at [this link](https://github.com/Jean2002mug/Data-science-final-project/blob/main/Final_project.pdf).

## Data Cleaning
The dataset contained no null values, but there were 8,733 instances of zero values. These zeros could indicate missing data or actual values, requiring careful treatment. To confirm the absence of null values, I used the `isnull()` function, which returned `False` for all entries.

To minimize skewed results, I filtered out zeros when analyzing `statistics.total_reviews`, `statistics.average_rating`, and `daily_average_publisher_revenue`, as they could distort median, mode, and other statistical calculations. For other columns, I retained the original dataset to preserve completeness.

### Handling Outliers
Outliers can significantly impact statistical analysis. To detect them, I applied the interquartile range (IQR) method and visualized distributions using boxplots. Key observations:
- `daily_average_gross_sales` and `daily_average_units_sold` exhibited extreme values, indicating bestsellers that may distort averages.
- `statistics.sale_price` contained unusually high values, possibly representing premium books.

For robustness, I used median-based statistics and log transformation where necessary to mitigate outlier effects.

## Analysis
### Task 1: Identifying the Genre with the Highest Averages
Using Python, I analyzed key metrics across genres:
- `daily_average_amazon_revenue`
- `statistics.average_rating`
- `statistics.total_reviews`
- `daily_average_gross_sales`
- `statistics.sale_price`
- `daily_average_units_sold`

For `statistics.total_reviews` and `statistics.average_rating`, I filtered out zeros to prevent skewed results. I employed Python’s `groupby()` function and used `agg()` with NumPy to compute the necessary statistics.

#### Results
- **Highest `statistics.sale_price`**: Nonfiction (**mean: 8.6**)
- **Highest `daily_average_amazon_revenue`**: Nonfiction (**mean: 76.99**)
- **Highest `statistics.total_reviews`**: Fiction (**mean: 332.7**)
- **Highest `daily_average_gross_sales`**: Fiction (**mean: 285.29**)
- **Highest `daily_average_units_sold`**: Fiction (**mean: 62.14**)
- **Highest `statistics.average_rating`**: Children (**mean: 4.5**)

#### Interpretation
- Fiction dominates in total reviews and gross sales, highlighting its widespread appeal.
- Nonfiction books command higher sale prices and generate the highest Amazon revenue.
- Children's books receive the highest ratings, suggesting strong reader satisfaction.
- Fiction’s high sales volume despite a lower price suggests affordability is a key factor in its popularity.

### Task 2: Relationship Between Publisher Type and Daily Units Sold
To examine the influence of publisher type on book sales, I:
1. Encoded `publisher_type` into dummy variables (Big Five, Single Author, Small/Medium, Amazon, Indie).
2. Appended these variables to the dataset.
3. Computed correlation coefficients between `daily_average_units_sold` and publisher types.
4. Visualized data using a scatterplot.

#### Results
- **Correlation coefficients:**
  - **Big Five**: 0.051
  - **Single Author**: -0.025
  - **Small/Medium**: -0.072
  - **Amazon**: 0.084
  - **Indie**: 0.024
- The scatterplot exhibited no clear pattern, reinforcing the weak relationship.

#### Interpretation
Correlation values close to zero indicate a weak relationship between publisher type and daily units sold. However, this does not rule out the impact of publisher type on other metrics, such as gross sales and ratings, warranting further analysis.

### Task 3: Most Popular Genre Among Publishers
To identify the most commonly published genre, I applied `value_counts()` on the `genre` column and extracted the top genre.

#### Result
- **Most published genre:** Fiction

### Task 4: Regression Analysis for Genre Impact on Revenue
To quantify the impact of genre on revenue, I performed multiple regression analysis using `daily_average_amazon_revenue` as the dependent variable and genres as categorical predictors. Results:
- Fiction and Nonfiction genres showed statistically significant coefficients, suggesting strong predictive power for revenue.
- Other genres had weaker influence, indicating niche markets.

### Task 5: Revenue Distribution Between Sellers, Publishers, and Authors
To assess revenue distribution among sellers, publishers, and authors, I:
1. Calculated total daily seller revenue.
2. Computed mean revenues for sellers, publishers, and authors.
3. Compared values using NumPy.
4. Visualized revenue distribution using a bar graph.

#### Results
- **Mean daily seller revenue:** 79.4
- **Mean daily publisher revenue:** 91.72
- **Mean daily author revenue:** Lower than both seller and publisher revenue.
- **Revenue Breakdown:**
  - Amazon takes a significant share from each sale.
  - Publishers retain a large portion of profits.
  - Authors earn the smallest share, raising concerns about fair compensation.

The bar graph illustrated that publishers and sellers earned significantly more than authors.

#### Interpretation
Sellers and publishers reap the highest revenue, while authors earn significantly less. Additionally, larger confidence intervals in seller and publisher revenue suggest considerable variability in their earnings. This finding underscores the economic imbalance in the book industry, where distribution and marketing play a crucial role in revenue allocation.

## Conclusion
This analysis provides valuable insights into book sales and publisher influence. Fiction remains the most popular genre, while Nonfiction generates the highest revenue. Publisher type exhibits a weak correlation with daily units sold, and revenue distribution heavily favors sellers and publishers over authors. Regression analysis confirmed that Fiction and Nonfiction significantly impact revenue, while other genres contribute less. Further exploration is recommended to determine additional factors influencing book sales performance.

## Sources
- Data source: [Book Sales Dataset](https://github.com/Jean2002mug/Data-science-final-project/blob/main/Final_project.pdf)
- Python libraries used: Pandas, NumPy, Matplotlib, Scikit-learn
- Statistical concepts: Mean, Correlation Coefficients, Regression Analysis, Data Cleaning Best Practices

