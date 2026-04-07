from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def get_models():
    """
    Models used in research paper
    """
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(kernel='linear'),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier()
    }