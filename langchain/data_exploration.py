import pandas as pd

data = pd.read_pickle('../data/course_info.pkl')
# data = pd.read_pickle('course_questions.pkl')

# data.info()
# print(data.head())
#
# for col in data.columns:
#     print(col + ': ' + str(data.iloc[0][col]))

# overview_objectives:
# overview_learning_outcomes:
# overview_description.en:

for col in ["title_en", "overview_objectives", "overview_learning_outcomes", "overview_description.en"]:
    print(col + ': ' + str(data[col].dtypes))

for col in ["title_en", "overview_objectives", "overview_learning_outcomes", "overview_description.en"]:
    print(col + ': ' + str(data.iloc[0][col]))
