import joblib
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# 1. Geração da massa de dados sintética
X, y = make_classification(
    n_samples=500,        # Total de exemplos
    n_features=4,         # Número de atributos (variáveis continuas)
    n_informative=3,      # Atributos com relevância para a predição
    n_redundant=1,        # Atributos redundantes
    n_classes=2,          # Classificação binária (0 ou 1)
    random_state=42
)

# 2. Divisão entre treino (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Inicialização e treinamento do modelo Naïve Bayes Gaussiano
model = GaussianNB()
model.fit(X_train, y_train)

# 4. Avaliação do modelo
y_pred = model.predict(X_test)

print("=== Nível Básico: Gaussian Naïve Bayes ===")
print(f"Acurácia: {accuracy_score(y_test, y_pred):.2f}\n")
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))

# 5. Salvar o modelo em um arquivo .pkl
joblib.dump(model, 'modelo_nb1.pkl')
print("\nModelo salvo como 'modelo_nb1.pkl'")

# 6. Mostrar Matriz de Confusão e Curva ROC
# Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
disp_cm = ConfusionMatrixDisplay(confusion_matrix=cm)
disp_cm.plot(cmap='Blues')
plt.title('Matriz de Confusão - Gaussian NB')
plt.savefig('cm_nb1.png')
print("Gráfico da Matriz de Confusão salvo como 'cm_nb1.png'")
plt.close()

# Curva ROC
y_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
disp_roc = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
disp_roc.plot()
plt.title('Curva ROC - Gaussian NB')
plt.savefig('roc_nb1.png')
print("Gráfico da Curva ROC salvo como 'roc_nb1.png'")
plt.close()