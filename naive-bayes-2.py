import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

# 1. Geração de massa de dados categóricos estruturados (Pandas Dataframe)
np.random.seed(42)
n_samples = 1000

escolaridade = np.random.choice(['Medio', 'Superior', 'Pos-Graduacao'], size=n_samples, p=[0.4, 0.4, 0.2])
renda_faixa = np.random.choice(['Baixa', 'Media', 'Alta'], size=n_samples, p=[0.3, 0.5, 0.2])
historico_credito = np.random.choice(['Ruim', 'Bom', 'Excelente'], size=n_samples, p=[0.2, 0.6, 0.2])

perfil_risco = []
for esc, ren, hist in zip(escolaridade, renda_faixa, historico_credito):
    score = 0
    if esc == 'Medio': score += 1
    if ren == 'Baixa': score += 2
    if hist == 'Ruim': score += 3
    if hist == 'Bom': score -= 1
    if hist == 'Excelente': score -= 2
    
    # Probabilidade de ser risco 'Alto' aumenta conforme o score de problemas
    prob_alto = min(max(0.05, score * 0.15 + 0.1), 0.95) 
    
    if np.random.rand() < prob_alto:
        perfil_risco.append('Alto')
    else:
        perfil_risco.append('Baixo')

dados = pd.DataFrame({
    'escolaridade': escolaridade,
    'renda_faixa': renda_faixa,
    'historico_credito': historico_credito,
    'perfil_risco': perfil_risco
})

# Separação de features (X) e rótulo (y)
X = dados[['escolaridade', 'renda_faixa', 'historico_credito']]
y = dados['perfil_risco']

categorical_cols = X.columns.tolist()

# 2. Pré-processamento e construção da Pipeline
# O CategoricalNB exige dados codificados numericamente para categorias
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols)
    ]
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', CategoricalNB())
])

# 3. Validação Cruzada (5-folds) para verificar estabilidade do modelo
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')

# 4. Divisão Treino/Teste e Ajuste Final
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
pipeline.fit(X_train, y_train)

# 5. Avaliação do Modelo
y_pred = pipeline.predict(X_test)

print("=== Nível Intermediário: Categorical Naïve Bayes ===")
print(f"Média Acurácia (Cross-Validation 5-Fold): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})\n")
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# 6. Salvar o modelo em um arquivo .pkl
joblib.dump(pipeline, 'modelo_nb2.pkl')
print("\nModelo salvo como 'modelo_nb2.pkl'")

# 7. Mostrar Matriz de Confusão e Curva ROC
# Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
disp_cm = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipeline.classes_)
disp_cm.plot(cmap='Blues')
plt.title('Matriz de Confusão - Categorical NB')
plt.savefig('cm_nb2.png')
print("Gráfico da Matriz de Confusão salvo como 'cm_nb2.png'")
plt.close()

# Curva ROC
pos_label = pipeline.classes_[1]
y_proba = pipeline.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba, pos_label=pos_label)
roc_auc = auc(fpr, tpr)
disp_roc = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
disp_roc.plot()
plt.title('Curva ROC - Categorical NB')
plt.savefig('roc_nb2.png')
print("Gráfico da Curva ROC salvo como 'roc_nb2.png'")
plt.close()