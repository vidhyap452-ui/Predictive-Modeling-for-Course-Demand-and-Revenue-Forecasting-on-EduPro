import pandas as pd
import numpy as np

np.random.seed(42)

categories = ["Data Science", "Web Development", "Business", "Design", "Marketing", "Cybersecurity", "AI & ML", "Finance"]
levels = ["Beginner", "Intermediate", "Advanced"]
course_types = ["Video", "Live", "Hybrid"]
expertise_list = ["Python", "JavaScript", "Finance", "UX/UI", "SEO", "Network Security", "Machine Learning", "Accounting"]

N_COURSES = 120
N_TEACHERS = 40
N_TRANSACTIONS = 2000

# ── Teachers ──────────────────────────────────────────────────────────────────
teacher_ids = [f"T{str(i).zfill(3)}" for i in range(1, N_TEACHERS + 1)]
teachers_df = pd.DataFrame({
    "TeacherID": teacher_ids,
    "Expertise": np.random.choice(expertise_list, N_TEACHERS),
    "YearsOfExperience": np.random.randint(1, 20, N_TEACHERS),
    "TeacherRating": np.round(np.random.uniform(3.0, 5.0, N_TEACHERS), 1),
})

# ── Courses ───────────────────────────────────────────────────────────────────
course_ids = [f"C{str(i).zfill(3)}" for i in range(1, N_COURSES + 1)]
course_categories = np.random.choice(categories, N_COURSES)
course_levels = np.random.choice(levels, N_COURSES)
course_prices = np.where(
    course_levels == "Beginner", np.random.uniform(9, 49, N_COURSES),
    np.where(course_levels == "Intermediate", np.random.uniform(49, 99, N_COURSES),
             np.random.uniform(99, 199, N_COURSES))
)

courses_df = pd.DataFrame({
    "CourseID": course_ids,
    "CourseCategory": course_categories,
    "CourseType": np.random.choice(course_types, N_COURSES),
    "CourseLevel": course_levels,
    "CoursePrice": np.round(course_prices, 2),
    "CourseDuration": np.random.randint(4, 60, N_COURSES),  # hours
    "CourseRating": np.round(np.random.uniform(3.2, 5.0, N_COURSES), 1),
    "TeacherID": np.random.choice(teacher_ids, N_COURSES),
})

# ── Transactions ──────────────────────────────────────────────────────────────
dates = pd.date_range("2022-01-01", "2024-12-31", periods=N_TRANSACTIONS)
transaction_course_ids = np.random.choice(course_ids, N_TRANSACTIONS,
    p=np.random.dirichlet(np.ones(N_COURSES) * 0.5))  # skewed popularity

txn_prices = []
for cid in transaction_course_ids:
    base = float(courses_df.loc[courses_df["CourseID"] == cid, "CoursePrice"].values[0])
    txn_prices.append(round(base * np.random.uniform(0.85, 1.0), 2))

transactions_df = pd.DataFrame({
    "TransactionID": [f"TXN{str(i).zfill(5)}" for i in range(1, N_TRANSACTIONS + 1)],
    "CourseID": transaction_course_ids,
    "TransactionDate": dates,
    "Amount": txn_prices,
})

def get_merged():
    """Return a fully merged and feature-engineered dataframe."""
    # Aggregate transactions → per course
    agg = transactions_df.groupby("CourseID").agg(
        EnrollmentCount=("TransactionID", "count"),
        TotalRevenue=("Amount", "sum"),
        AvgRevenue=("Amount", "mean"),
    ).reset_index()

    df = courses_df.merge(agg, on="CourseID", how="left").fillna(0)
    df = df.merge(teachers_df, on="TeacherID", how="left")

    # Revenue per enrollment
    df["RevenuePerEnrollment"] = np.where(
        df["EnrollmentCount"] > 0, df["TotalRevenue"] / df["EnrollmentCount"], 0)

    # Price bands
    df["PriceBand"] = pd.cut(df["CoursePrice"], bins=[0, 49, 99, 999],
                              labels=["Low", "Medium", "High"])

    # Duration buckets
    df["DurationBucket"] = pd.cut(df["CourseDuration"], bins=[0, 10, 30, 999],
                                   labels=["Short", "Medium", "Long"])

    # Rating tier
    df["RatingTier"] = pd.cut(df["CourseRating"], bins=[0, 3.5, 4.2, 5.0],
                               labels=["Low", "Medium", "High"])

    # Experience buckets
    df["ExperienceBucket"] = pd.cut(df["YearsOfExperience"], bins=[0, 5, 10, 99],
                                     labels=["Junior", "Mid", "Senior"])

    return df

def get_monthly_revenue():
    merged = transactions_df.merge(courses_df[["CourseID", "CourseCategory"]], on="CourseID")
    merged["Month"] = pd.to_datetime(merged["TransactionDate"]).dt.to_period("M").astype(str)
    return merged.groupby(["Month", "CourseCategory"])["Amount"].sum().reset_index()

def get_courses():
    return courses_df

def get_teachers():
    return teachers_df

def get_transactions():
    return transactions_df
