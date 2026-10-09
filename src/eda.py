import os
from dotenv import load_dotenv
import snowflake.connector
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

load_dotenv()

conn = snowflake.connector.connect(
    user=os.getenv("SNOWFLAKE_USER"),
    password=os.getenv("SNOWFLAKE_PASSWORD"),
    account=os.getenv("SNOWFLAKE_ACCOUNT"),
    warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
    database=os.getenv("SNOWFLAKE_DATABASE"),
    schema=os.getenv("SNOWFLAKE_SCHEMA"),
)
df = pd.read_sql("SELECT * FROM hr_attrition_final;", conn)
conn.close()

# 1. Overall attrition rate
print("Overall Attrition Rate:")
print(df['ATTRITION'].value_counts(normalize=True) * 100)

# 2. Attrition by department
plt.figure(figsize=(8,5))
sns.countplot(data=df, x='DEPARTMENT', hue='ATTRITION')
plt.title("Attrition by Department")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/attrition_by_department.png")
plt.close()

# 3. Attrition by overtime
plt.figure(figsize=(6,5))
sns.countplot(data=df, x='OVERTIME', hue='ATTRITION')
plt.title("Attrition by Overtime")
plt.tight_layout()
plt.savefig("outputs/attrition_by_overtime.png")
plt.close()

# 4. Salary vs Attrition
plt.figure(figsize=(6,5))
sns.boxplot(data=df, x='ATTRITION', y='MONTHLY_INCOME')
plt.title("Monthly Income vs Attrition")
plt.tight_layout()
plt.savefig("outputs/salary_vs_attrition.png")
plt.close()

print("EDA charts saved: attrition_by_department.png, attrition_by_overtime.png, salary_vs_attrition.png")