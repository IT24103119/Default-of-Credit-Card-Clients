# =============================================================
# IT3081(SM) Consultancy Project — Credit Card Default Analysis
# =============================================================

# =============================================================
# Script 01 – Setup & Initial Data Exploration
# -------------------------------------------------------------
# 1.Installed and loaded required R packages — dplyr and ggplot2 for data manipulation and visualization, corrplot for correlation plotting (along with glmnet, leaps, naniar, car for later tasks).
# 
# 2.Loaded the dataset into R using read.csv(), importing all 30,000 records and 25 variables from the UCI Credit Card Default dataset.
# 
# 3.Inspected the dataset structure using dim(), str(), head(), and colnames() to confirm the number of observations, variable types, and column names.
# 
# 4.Checked data quality by verifying there were no missing values (sum(is.na(df))) and no duplicate rows (sum(duplicated(df))) — both returned 0, confirming the dataset is clean.
# 
# 5.Renamed the target variable from default.payment.next.month to default for easier reference in code.
# 
# 6.Converted categorical variables to factors — SEX (Male/Female) and default (No Default/Default) were labeled for clearer output and plotting. 
# 
# 7.EDUCATION and MARRIAGE were checked using table(), revealing undocumented category codes (0, 5, 6 in EDUCATION; 0 in MARRIAGE) that don't match the official coding scheme — flagged as a data quality issue.
# 
# 8.Examined the target variable balance — found a 77.9% / 22.1% split between non-defaulters and defaulters, indicating a moderately imbalanced target that should be considered during model evaluation in  Task 5.
# 
# 9.Produced summary statistics for all variables using summary() to get an initial sense of distributions and ranges.
# 
# 10.Generated a correlation matrix and heatmap across the six BILL_AMT and six PAY_AMT variables to check for multicollinearity — this provides early evidence for the Task 7 discussion on whether PCA would benefit this dataset.
# =============================================================

# ---- 1. Install & load required packages (run once) ----
required_packages <- c("dplyr", "ggplot2", "glmnet", "leaps", "naniar", "car", "corrplot")

installed <- rownames(installed.packages())
to_install <- setdiff(required_packages, installed)
if (length(to_install) > 0) install.packages(to_install)

library(dplyr)
library(ggplot2)
library(corrplot)

# ---- 2. Load dataset ----
# Update the path to wherever you save the CSV on your machine
df <- read.csv("C:/Users/chris/OneDrive - Sri Lanka Institute of Information Technology/Uni/Y3S1/SM/Assignment/SM 2026-DS-88/UCI_Credit_Card/UCI_Credit_Card.csv", stringsAsFactors = FALSE)

# ---- 3. First look ----
dim(df)                           # rows x columns
str(df)                           # variable types
head(df)                          # first few rows
colnames(df)

# ---- 4. Data quality checks ----
sum(is.na(df))                    # total missing values (expect 0)
sum(duplicated(df))               # duplicate rows (expect 0)

# ---- 5. Rename target variable for convenience ----
df <- df %>% rename(default = default.payment.next.month)

# ---- 6. Convert categorical variables to factors ----
df$SEX <- factor(df$SEX, labels = c("Male", "Female"))
df$default <- factor(df$default, labels = c("No Default", "Default"))

# EDUCATION and MARRIAGE have some undocumented codes (0, 5, 6 in EDUCATION; 0 in MARRIAGE) — flag these for your data quality section, don't just drop them silently
table(df$EDUCATION)
table(df$MARRIAGE)

# ---- 7. Target variable balance ----
table(df$default)
prop.table(table(df$default))     # ~78% / 22% split — note this for Task 3 & 5

ggplot(df, aes(x = default, fill = default)) +
  geom_bar() +
  labs(title = "Default vs Non-Default Distribution",
       x = "Default Status", y = "Count") +
  theme_minimal()

# ---- 8. Quick summary stats ----
summary(df)

# ---- 9. Correlation check among BILL_AMT / PAY_AMT variables ----
# (useful evidence for Task 7's PCA discussion — these tend to be highly correlated)
bill_pay_vars <- df %>%
  select(BILL_AMT1:BILL_AMT6, PAY_AMT1:PAY_AMT6)

corr_matrix <- cor(bill_pay_vars)
corrplot(corr_matrix, method = "color", type = "upper",
         tl.cex = 0.7, title = "Correlation: Bill & Payment Amounts",
         mar = c(0,0,1,0))



# =============================================================
# Script 02 – Descriptive Analysis (Task 3)
# -------------------------------------------------------------
# 1.Recoded EDUCATION and MARRIAGE into readable labels — official categories (Graduate School, University, High School, Others for EDUCATION; Married, Single, Others for MARRIAGE) were labeled clearly, while undocumented codes were grouped into "Unknown/Undocumented" rather than dropped, preserving all 30,000 records.
# 
# 2.Quantified the data quality issue — found 345 rows (1.15%) with undocumented EDUCATION codes and 54 rows (0.18%) with undocumented MARRIAGE codes. Both are small enough not to materially bias the analysis but are noted as a data quality limitation.
# 
# 3.Detected outliers using the IQR method on four key continuous variables:
#   LIMIT_BAL: 167 outliers (0.56%)
#   AGE: 272 outliers (0.91%)
#   BILL_AMT1: 2,400 outliers (8.00%)
#   PAY_AMT1: 2,745 outliers (9.15%)
#   The higher outlier rates in BILL_AMT1 and PAY_AMT1 reflect the natural right-skew of financial transaction data rather than data entry errors, so they were retained rather than removed.
# 
# 4.Visualized univariate distributions for AGE, LIMIT_BAL, EDUCATION, and MARRIAGE using histograms and bar charts to understand the overall client profile.
# 
# 5.Analyzed default rate by sex — found males default at a higher rate (24.2%) than females (20.8%).
# 
# 6.Analyzed default rate by education level — found an inverse relationship with education: High School clients had the highest default rate (25.2%), followed by University (23.7%) and Graduate School (19.2%), suggesting education level is associated with credit risk.
# 
# 7.Visualized default rate by marital status, age, and credit limit using proportion bar charts and boxplots to compare defaulters against non-defaulters across these dimensions.
# 
# 8.Examined default rate by repayment status (PAY_0) — the most recent month's repayment status, to identify whether payment delay history is a strong indicator of future default (this variable typically shows the sharpest relationship with the target).
# 
# 9.Generated a full correlation heatmap across all numeric predictors (credit limit, age, repayment status, bill amounts, payment amounts) to detect multicollinearity — reinforcing the earlier evidence for the Task 7 PCA discussion.
# 
# 10.Produced a summary comparison table between defaulters and non-defaulters: defaulters had a slightly higher average age (35.7 vs 35.4), a notably lower average credit limit (NT$130,110 vs NT$178,100), and lower average payment amounts (NT$3,397 vs NT$6,307) — indicating credit limit and repayment behavior are meaningful early predictors of default risk.
# =============================================================


library(dplyr)
library(ggplot2)
library(corrplot)

# ---- 0. Reload & prep (skip if df already in your environment from Script 01) ----
# df <- read.csv("UCI_Credit_Card.csv", stringsAsFactors = FALSE)
# df <- df %>% rename(default = default.payment.next.month)
# df$SEX <- factor(df$SEX, labels = c("Male", "Female"))
# df$default <- factor(df$default, labels = c("No Default", "Default"))

# ---- 1. Recode EDUCATION and MARRIAGE for readable labels ----
# Official codes: EDUCATION 1=grad school,2=university,3=high school,4=others
# 0,5,6 are undocumented — grouped into "Other/Unknown" rather than dropped
df$EDUCATION_LABEL <- case_when(
  df$EDUCATION == 1 ~ "Graduate School",
  df$EDUCATION == 2 ~ "University",
  df$EDUCATION == 3 ~ "High School",
  df$EDUCATION == 4 ~ "Others",
  TRUE ~ "Unknown/Undocumented"     # covers 0, 5, 6
)
df$EDUCATION_LABEL <- factor(df$EDUCATION_LABEL)

# MARRIAGE: 1=married,2=single,3=others, 0=undocumented
df$MARRIAGE_LABEL <- case_when(
  df$MARRIAGE == 1 ~ "Married",
  df$MARRIAGE == 2 ~ "Single",
  df$MARRIAGE == 3 ~ "Others",
  TRUE ~ "Unknown/Undocumented"     # covers 0
)
df$MARRIAGE_LABEL <- factor(df$MARRIAGE_LABEL)

# ---- 2. Data quality summary table (put this in your report) ----
cat("Undocumented EDUCATION codes:", sum(df$EDUCATION_LABEL == "Unknown/Undocumented"),
    "rows (", round(100*sum(df$EDUCATION_LABEL == "Unknown/Undocumented")/nrow(df), 2), "%)\n")
cat("Undocumented MARRIAGE codes:", sum(df$MARRIAGE_LABEL == "Unknown/Undocumented"),
    "rows (", round(100*sum(df$MARRIAGE_LABEL == "Unknown/Undocumented")/nrow(df), 2), "%)\n")

# ---- 3. Outlier detection (IQR method) on key continuous variables ----
detect_outliers <- function(x, varname) {
  Q1 <- quantile(x, 0.25); Q3 <- quantile(x, 0.75); IQR_val <- Q3 - Q1
  lower <- Q1 - 1.5 * IQR_val
  upper <- Q3 + 1.5 * IQR_val
  n_out <- sum(x < lower | x > upper)
  cat(sprintf("%-12s : %d outliers (%.2f%%) | bounds [%.0f, %.0f]\n",
              varname, n_out, 100*n_out/length(x), lower, upper))
}

cat("\n--- Outlier Summary (IQR method) ---\n")
detect_outliers(df$LIMIT_BAL, "LIMIT_BAL")
detect_outliers(df$AGE, "AGE")
detect_outliers(df$BILL_AMT1, "BILL_AMT1")
detect_outliers(df$PAY_AMT1, "PAY_AMT1")

# ---- 4. Univariate visualizations ----

# Age distribution
ggplot(df, aes(x = AGE)) +
  geom_histogram(binwidth = 5, fill = "#2c7fb8", color = "white") +
  labs(title = "Age Distribution of Credit Card Clients", x = "Age", y = "Count") +
  theme_minimal()

# Credit limit distribution (log scale since it's right-skewed)
ggplot(df, aes(x = LIMIT_BAL)) +
  geom_histogram(bins = 40, fill = "#41ab5d", color = "white") +
  scale_x_continuous(labels = scales::comma) +
  labs(title = "Credit Limit Distribution", x = "Credit Limit (NT$)", y = "Count") +
  theme_minimal()

# Education level
ggplot(df, aes(x = EDUCATION_LABEL, fill = EDUCATION_LABEL)) +
  geom_bar() +
  labs(title = "Education Level Distribution", x = "Education", y = "Count") +
  theme_minimal() +
  theme(legend.position = "none", axis.text.x = element_text(angle = 30, hjust = 1))

# Marital status
ggplot(df, aes(x = MARRIAGE_LABEL, fill = MARRIAGE_LABEL)) +
  geom_bar() +
  labs(title = "Marital Status Distribution", x = "Marital Status", y = "Count") +
  theme_minimal() +
  theme(legend.position = "none")

# ---- 5. Bivariate: default rate by demographic groups ----

# Default rate by sex
df %>%
  group_by(SEX) %>%
  summarise(default_rate = mean(default == "Default"), n = n())

ggplot(df, aes(x = SEX, fill = default)) +
  geom_bar(position = "fill") +
  labs(title = "Default Rate by Sex", x = "Sex", y = "Proportion") +
  scale_y_continuous(labels = scales::percent) +
  theme_minimal()

# Default rate by education
df %>%
  group_by(EDUCATION_LABEL) %>%
  summarise(default_rate = mean(default == "Default"), n = n()) %>%
  arrange(desc(default_rate))

ggplot(df, aes(x = EDUCATION_LABEL, fill = default)) +
  geom_bar(position = "fill") +
  labs(title = "Default Rate by Education Level", x = "Education", y = "Proportion") +
  scale_y_continuous(labels = scales::percent) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

# Default rate by marital status
ggplot(df, aes(x = MARRIAGE_LABEL, fill = default)) +
  geom_bar(position = "fill") +
  labs(title = "Default Rate by Marital Status", x = "Marital Status", y = "Proportion") +
  scale_y_continuous(labels = scales::percent) +
  theme_minimal()

# Age vs default (boxplot)
ggplot(df, aes(x = default, y = AGE, fill = default)) +
  geom_boxplot() +
  labs(title = "Age Distribution by Default Status", x = "Default Status", y = "Age") +
  theme_minimal() +
  theme(legend.position = "none")

# Credit limit vs default (boxplot)
ggplot(df, aes(x = default, y = LIMIT_BAL, fill = default)) +
  geom_boxplot() +
  scale_y_continuous(labels = scales::comma) +
  labs(title = "Credit Limit by Default Status", x = "Default Status", y = "Credit Limit (NT$)") +
  theme_minimal() +
  theme(legend.position = "none")

# ---- 6. Repayment status pattern (PAY_0 is most recent month) ----
ggplot(df, aes(x = factor(PAY_0), fill = default)) +
  geom_bar(position = "fill") +
  labs(title = "Default Rate by Most Recent Repayment Status (PAY_0)",
       x = "Repayment Status Code (PAY_0)", y = "Proportion",
       subtitle = "-2/-1/0 = paid on time or ahead; 1+ = months delayed") +
  scale_y_continuous(labels = scales::percent) +
  theme_minimal()

# ---- 7. Correlation heatmap across all numeric variables (excluding ID) ----
numeric_vars <- df %>%
  select(LIMIT_BAL, AGE, PAY_0:PAY_6, BILL_AMT1:BILL_AMT6, PAY_AMT1:PAY_AMT6)

corr_all <- cor(numeric_vars)
corrplot(corr_all, method = "color", type = "upper", tl.cex = 0.6,
         title = "Full Correlation Matrix", mar = c(0,0,1,0))

# ---- 8. Summary table for report ----
summary_table <- df %>%
  group_by(default) %>%
  summarise(
    n = n(),
    mean_age = round(mean(AGE), 1),
    mean_limit_bal = round(mean(LIMIT_BAL), 0),
    mean_bill_amt1 = round(mean(BILL_AMT1), 0),
    mean_pay_amt1 = round(mean(PAY_AMT1), 0)
  )
print(summary_table)



# =============================================================
# Script 03 – Statistical Inference (Task 4)
# -------------------------------------------------------------
# 1.Tested default rate by sex using a two-proportion z-test (prop.test) — found males default at 24.2% vs females at 20.8%, a statistically significant difference (p < 0.001).

# 2.Compared mean credit limit between defaulters and non-defaulters using Welch's t-test — non-defaulters averaged NT$178,100 vs defaulters at NT$130,110, a highly significant difference (p < 0.001), confirming credit limit as a meaningful predictor.

# 3.Compared mean most-recent payment amount (PAY_AMT1) between groups using Welch's t-test — non-defaulters paid NT$6,307 on average vs defaulters at NT$3,397 (p < 0.001), also highly significant.

# 4.Tested variance of BILL_AMT1 between default groups using both an F-test and Levene's test — the F-test showed no significant variance difference (p = 0.775), but Levene's test (more robust to non-normal data) found a significant difference (p < 0.001), so the two methods disagreed and this is worth noting as a limitation.

# 5.Ran a one-way ANOVA comparing mean credit limit across education levels — found a highly significant difference (F = 581.2, p < 0.001). A Tukey HSD post-hoc test showed Graduate School clients have significantly higher credit limits than High School and University clients.

# 6.Ran a one-way ANOVA comparing mean credit limit across marital status — also highly significant (F = 131.1, p < 0.001), with Married clients holding notably higher limits than Single clients.

# 7.Tested whether default rate is independent of education level using a chi-square test of independence — found a significant association (χ² = 160.59, p < 0.001), confirming education level and default risk are related, consistent with the descriptive pattern from Script 02.
# =============================================================


library(dplyr)
library(car)

# =============================================================
# TEST 1 — Comparison of Proportions: Default rate by SEX
# =============================================================
# H0: Default proportion is equal between males and females
# H1: Default proportion differs between males and females
# Why this test: comparing a binary outcome (default) across 2 groups → prop.test

sex_table <- table(df$SEX, df$default)
print(sex_table)

prop_test_sex <- prop.test(sex_table[, "Default"], rowSums(sex_table))
print(prop_test_sex)

# INTERPRET: check p-value against alpha = 0.05.
# From Script 02: Male 24.2% vs Female 20.8% default rate — likely significant given n=30,000.
# Business implication: if significant, sex could be a factor in risk scoring
# (note: using SEX in credit risk models raises fairness/ethical considerations —
# worth flagging in Task 12's ethical considerations section)


# =============================================================
# TEST 2 — Comparison of Means: LIMIT_BAL by default status
# =============================================================
# H0: Mean credit limit is equal for defaulters and non-defaulters
# H1: Mean credit limit differs between defaulters and non-defaulters
# Why this test: comparing a continuous variable's mean across 2 groups → t-test
# (Welch's t-test used by default in R — robust to unequal variances)

t_test_limit <- t.test(LIMIT_BAL ~ default, data = df)
print(t_test_limit)

# INTERPRET: From Script 02, non-defaulters averaged NT$178,100 vs defaulters NT$130,110.
# A significant result supports credit limit as a meaningful predictor for Task 5.


# =============================================================
# TEST 3 — Comparison of Means: PAY_AMT1 by default status
# =============================================================
# H0: Mean most-recent payment amount is equal between groups
# H1: Mean most-recent payment amount differs between groups

t_test_payamt <- t.test(PAY_AMT1 ~ default, data = df)
print(t_test_payamt)


# =============================================================
# TEST 4 — Comparison of Variances: BILL_AMT1 by default status
# =============================================================
# H0: Variance of BILL_AMT1 is equal between defaulters and non-defaulters
# H1: Variances differ between groups
# Why this test: checks whether bill amount variability differs by risk group —
# also informs which t-test assumption (equal/unequal variance) is appropriate

var_test_bill <- var.test(BILL_AMT1 ~ default, data = df)
print(var_test_bill)

# Levene's test as a more robust alternative (less sensitive to non-normality)
leveneTest(BILL_AMT1 ~ default, data = df)


# =============================================================
# TEST 5 — ANOVA: LIMIT_BAL across EDUCATION levels
# =============================================================
# H0: Mean credit limit is equal across all education levels
# H1: At least one education level has a different mean credit limit
# Why this test: comparing a continuous variable's mean across MORE than 2 groups → ANOVA

anova_education <- aov(LIMIT_BAL ~ EDUCATION_LABEL, data = df)
summary(anova_education)

# If significant, run a post-hoc test to see WHICH groups differ
TukeyHSD(anova_education)

# Visual check of assumption (roughly equal spread across groups)
boxplot(LIMIT_BAL ~ EDUCATION_LABEL, data = df,
        main = "Credit Limit by Education Level", ylab = "Credit Limit (NT$)")


# =============================================================
# TEST 6 — ANOVA: LIMIT_BAL across MARRIAGE status
# =============================================================
# H0: Mean credit limit is equal across marital status groups
# H1: At least one marital status group has a different mean credit limit

anova_marriage <- aov(LIMIT_BAL ~ MARRIAGE_LABEL, data = df)
summary(anova_marriage)
TukeyHSD(anova_marriage)


# =============================================================
# TEST 7 — Comparison of Proportions: Default rate across EDUCATION (chi-square)
# =============================================================
# H0: Default rate is independent of education level
# H1: Default rate depends on education level
# Why this test: comparing a categorical outcome across MORE than 2 categorical
# groups → chi-square test of independence (extension of prop.test for >2 groups)

edu_table <- table(df$EDUCATION_LABEL, df$default)
print(edu_table)
chisq_edu <- chisq.test(edu_table)
print(chisq_edu)




# =============================================================
# Script 04 – Predictive Statistical Modelling (Task 5)
# -------------------------------------------------------------
# 1.Prepared the modelling dataset by removing the ID column and the label columns used only for plotting, then split the data into 80% training and 20% test sets for honest model evaluation.

# 2.Built a baseline logistic regression model using all predictors, since the target variable (default) is binary — the natural fit for this problem.

# 3.Checked for multicollinearity using Variance Inflation Factor (VIF) — [fill in: which variables showed high VIF, e.g. BILL_AMT columns] — this justifies trying regularized regression methods next.

# 4.Performed stepwise variable selection — both backward elimination (starting from the full model) and forward selection (starting from no predictors) — to identify a more parsimonious model.

# 5.Fitted Ridge, LASSO, and Elastic Net regression using cross-validation to select the optimal penalty strength for each — these methods handle multicollinearity better than standard logistic regression and LASSO additionally performs automatic variable selection.

# 6.Evaluated all five models (full logistic, backward selection, Ridge, LASSO, Elastic Net) on the held-out test set using Accuracy, Precision, Recall, and AUC — not accuracy alone, since the ~78/22 class imbalance means Recall and AUC better reflect real-world usefulness for a bank.

# 7.Compared model performance in a summary table and ROC curve plot — [fill in: which model had the best AUC/Recall] — this became the basis for recommending [fill in: chosen model] as the most suitable model for the bank's operational use.

# 8.Identified LASSO's variable shrinkage — [fill in: which variables were shrunk to exactly zero] — cross-checked against the significant predictors found in Script 03's inference tests.
# =============================================================

library(dplyr)
library(glmnet)
library(car)      # for VIF (multicollinearity check)
library(pROC)     # install.packages("pROC") if not already installed

set.seed(123)   # reproducibility

# =============================================================
# 1. PREPARE MODELLING DATASET
# =============================================================
# Drop ID (not predictive) and the character label columns (keep numeric codes
# for regression; EDUCATION_LABEL/MARRIAGE_LABEL were just for Script 02 plots)

model_df <- df %>%
  select(-ID, -EDUCATION_LABEL, -MARRIAGE_LABEL)

# default is already a factor (No Default / Default) — glm needs 0/1 or factor
# with the FIRST level as the reference (non-event). Check order:
levels(model_df$default)   # should show "No Default" "Default" — Default = event

# ---- Train/test split (80/20) ----
n <- nrow(model_df)
train_idx <- sample(1:n, size = 0.8 * n)
train_data <- model_df[train_idx, ]
test_data  <- model_df[-train_idx, ]

cat("Train rows:", nrow(train_data), "| Test rows:", nrow(test_data), "\n")


# =============================================================
# 2. BASELINE MODEL — Full Logistic Regression (all predictors)
# =============================================================
model_full <- glm(default ~ ., data = train_data, family = binomial)
summary(model_full)

# ---- Check multicollinearity (assumption check) ----
# VIF > 5-10 suggests problematic multicollinearity — expect this among
# the 6 BILL_AMT and 6 PAY_AMT variables (ties into your Task 7 PCA discussion)
vif(model_full)


# =============================================================
# 3. STEPWISE SELECTION — Forward & Backward
# =============================================================
# Backward: start with full model, remove least useful predictors
model_backward <- step(model_full, direction = "backward", trace = 0)
summary(model_backward)

# Forward: start with intercept-only, add predictors one at a time
model_null <- glm(default ~ 1, data = train_data, family = binomial)
model_forward <- step(model_null,
                      scope = list(lower = model_null, upper = model_full),
                      direction = "forward", trace = 0)
summary(model_forward)


# =============================================================
# 4. REGULARIZED REGRESSION — Ridge, LASSO, Elastic Net
# =============================================================
# glmnet needs a matrix of predictors and a numeric/factor response
x_train <- model.matrix(default ~ . - 1, data = train_data)  # -1 drops intercept col
y_train <- train_data$default
x_test  <- model.matrix(default ~ . - 1, data = test_data)
y_test  <- test_data$default

# ---- Ridge (alpha = 0) ----
cv_ridge <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 0)
plot(cv_ridge)
best_lambda_ridge <- cv_ridge$lambda.min
model_ridge <- glmnet(x_train, y_train, family = "binomial",
                      alpha = 0, lambda = best_lambda_ridge)
coef(model_ridge)

# ---- LASSO (alpha = 1) ----
cv_lasso <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 1)
plot(cv_lasso)
best_lambda_lasso <- cv_lasso$lambda.min
model_lasso <- glmnet(x_train, y_train, family = "binomial",
                      alpha = 1, lambda = best_lambda_lasso)
coef(model_lasso)   # variables shrunk to exactly 0 were "dropped" by LASSO

# ---- Elastic Net (alpha = 0.5, a mix of Ridge + LASSO) ----
cv_enet <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 0.5)
best_lambda_enet <- cv_enet$lambda.min
model_enet <- glmnet(x_train, y_train, family = "binomial",
                     alpha = 0.5, lambda = best_lambda_enet)
coef(model_enet)


# =============================================================
# 5. MODEL EVALUATION — compare all models on the TEST set
# =============================================================

evaluate_model <- function(probs, actual, model_name) {
  pred_class <- ifelse(probs > 0.5, "Default", "No Default")
  pred_class <- factor(pred_class, levels = levels(actual))
  
  cm <- table(Predicted = pred_class, Actual = actual)
  accuracy <- sum(diag(cm)) / sum(cm)
  precision <- cm["Default", "Default"] / sum(cm["Default", ])
  recall <- cm["Default", "Default"] / sum(cm[, "Default"])
  roc_obj <- roc(actual, probs, levels = rev(levels(actual)), quiet = TRUE)
  auc_val <- auc(roc_obj)
  
  cat("\n===", model_name, "===\n")
  print(cm)
  cat(sprintf("Accuracy: %.3f | Precision: %.3f | Recall: %.3f | AUC: %.3f\n",
              accuracy, precision, recall, auc_val))
  return(list(cm = cm, accuracy = accuracy, precision = precision,
              recall = recall, auc = auc_val, roc = roc_obj))
}

# Full logistic model
probs_full <- predict(model_full, newdata = test_data, type = "response")
res_full <- evaluate_model(probs_full, test_data$default, "Full Logistic Regression")

# Backward-selected model
probs_backward <- predict(model_backward, newdata = test_data, type = "response")
res_backward <- evaluate_model(probs_backward, test_data$default, "Backward Selection")

# Ridge
probs_ridge <- predict(model_ridge, newx = x_test, type = "response")[, 1]
res_ridge <- evaluate_model(probs_ridge, y_test, "Ridge Regression")

# LASSO
probs_lasso <- predict(model_lasso, newx = x_test, type = "response")[, 1]
res_lasso <- evaluate_model(probs_lasso, y_test, "LASSO Regression")

# Elastic Net
probs_enet <- predict(model_enet, newx = x_test, type = "response")[, 1]
res_enet <- evaluate_model(probs_enet, y_test, "Elastic Net")


# =============================================================
# 6. COMPARISON TABLE (for your report)
# =============================================================
comparison <- data.frame(
  Model = c("Full Logistic", "Backward Selection", "Ridge", "LASSO", "Elastic Net"),
  Accuracy = c(res_full$accuracy, res_backward$accuracy, res_ridge$accuracy,
               res_lasso$accuracy, res_enet$accuracy),
  Precision = c(res_full$precision, res_backward$precision, res_ridge$precision,
                res_lasso$precision, res_enet$precision),
  Recall = c(res_full$recall, res_backward$recall, res_ridge$recall,
             res_lasso$recall, res_enet$recall),
  AUC = c(res_full$auc, res_backward$auc, res_ridge$auc,
          res_lasso$auc, res_enet$auc)
)
print(comparison)

# ---- ROC curve comparison plot ----
plot(res_full$roc, col = "blue", main = "ROC Curve Comparison")
plot(res_lasso$roc, col = "red", add = TRUE)
plot(res_ridge$roc, col = "green", add = TRUE)
legend("bottomright", legend = c("Full Logistic", "LASSO", "Ridge"),
       col = c("blue", "red", "green"), lwd = 2)
