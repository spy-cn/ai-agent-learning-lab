# 第一个 NLP 程序

从零开始完成一个完整的 NLP 文本分类项目——从原始文本到模型预测的全流程。

## 项目目标

```
构建一个电影评论情感分析系统:

输入: "This movie is absolutely fantastic! Best film I've ever seen."
输出: 积极 (Positive) ✓

输入: "Terrible plot, awful acting. Complete waste of time."
输出: 消极 (Negative) ✗

完整流程:
原始文本 → 分词 → 清洗 → 向量化 → 分类器 → 预测
```

## 完整代码

### Step 1: 数据准备

```python
# step1_data.py — 准备训练数据

import pandas as pd
import numpy as np

# === 模拟电影评论数据集 ===
# 实际项目可用 IMDB 数据集: 
# from datasets import load_dataset
# dataset = load_dataset("imdb")

data = {
    'text': [
        # 积极
        "This movie is absolutely fantastic! Best film I've ever seen.",
        "Amazing storyline and brilliant acting. Highly recommended!",
        "One of the best movies of the year. Loved every minute.",
        "Stunning visuals and great soundtrack. A masterpiece.",
        "Wonderful performance by the entire cast. Must watch!",
        "I really enjoyed this film. It was heartwarming and funny.",
        "The plot was engaging from start to finish. Excellent work.",
        "Beautiful cinematography and powerful storytelling.",
        "An instant classic. Will definitely watch again.",
        "Perfect blend of humor and drama. Five stars!",
        
        # 消极
        "Terrible plot, awful acting. Complete waste of time.",
        "Boring and predictable. One of the worst movies ever.",
        "The story makes no sense. Don't waste your money.",
        "Disappointing sequel. Nothing like the original.",
        "Poor direction and bad special effects. Avoid at all costs.",
        "I fell asleep during the movie. That bad.",
        "The acting was wooden and the dialogue was cringeworthy.",
        "Too long, too slow, and completely pointless.",
        "Worst film I've seen this year. Total disaster.",
        "The plot had more holes than Swiss cheese. Terrible.",
    ],
    'label': [1]*10 + [0]*10  # 1=积极, 0=消极
}

df = pd.DataFrame(data)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # 打乱

print(f"数据集大小: {len(df)}")
print(f"积极: {sum(df['label']==1)}, 消极: {sum(df['label']==0)}")
print(f"\n样本示例:")
print(df.head())
```

### Step 2: 文本预处理

```python
# step2_preprocess.py — 文本预处理管道

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# 下载 NLTK 数据 (首次运行)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class TextPreprocessor:
    """文本预处理器"""
    
    def __init__(self, language='english'):
        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text):
        """基础清洗"""
        # 转小写
        text = text.lower()
        # 移除 HTML 标签
        text = re.sub(r'<[^>]+>', '', text)
        # 移除 URL
        text = re.sub(r'http\S+|www\S+', '', text)
        # 移除特殊字符 (保留字母和空格)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # 移除多余空格
        text = ' '.join(text.split())
        return text
    
    def tokenize(self, text):
        """分词"""
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens):
        """移除停用词"""
        return [t for t in tokens if t not in self.stop_words]
    
    def lemmatize(self, tokens):
        """词形还原"""
        return [self.lemmatizer.lemmatize(t) for t in tokens]
    
    def preprocess(self, text):
        """完整预处理管道"""
        text = self.clean_text(text)
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize(tokens)
        return ' '.join(tokens)

# 测试预处理器
preprocessor = TextPreprocessor()

sample = "This movie is <b>FANTASTIC!!!</b> The acting was absolutely amazing."
processed = preprocessor.preprocess(sample)
print(f"原始: {sample}")
print(f"处理后: {processed}")
# 处理后: movie fantastic acting absolutely amazing

# 批量处理
df['processed_text'] = df['text'].apply(preprocessor.preprocess)
print(f"\n处理后的数据:")
print(df[['processed_text', 'label']].head())
```

### Step 3: 特征提取 (TF-IDF)

```python
# step3_features.py — TF-IDF 特征提取

from sklearn.feature_extraction.text import TfidfVectorizer

# 创建 TF-IDF 向量化器
vectorizer = TfidfVectorizer(
    max_features=5000,      # 最大特征数
    min_df=1,               # 最小文档频率
    max_df=0.9,             # 最大文档频率 (过滤过于常见的词)
    ngram_range=(1, 2),     # 使用 unigram + bigram
)

# 拟合和转换
X = vectorizer.fit_transform(df['processed_text'])
y = df['label'].values

print(f"特征矩阵形状: {X.shape}")
print(f"词汇表大小: {len(vectorizer.vocabulary_)}")

# 查看部分词汇
vocab_sample = list(vectorizer.vocabulary_.items())[:10]
print(f"\n词汇表示例:")
for word, idx in vocab_sample:
    print(f"  {word}: {idx}")

# 查看特征值
print(f"\n第一条文本的 TF-IDF 向量 (非零部分):")
feature_names = vectorizer.get_feature_names_out()
first_vec = X[0].toarray()[0]
non_zero = [(feature_names[i], first_vec[i]) 
            for i in range(len(first_vec)) if first_vec[i] > 0]
non_zero.sort(key=lambda x: x[1], reverse=True)
for word, score in non_zero[:5]:
    print(f"  {word}: {score:.4f}")
```

### Step 4: 训练分类器

```python
# step4_train.py — 训练情感分类器

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

# 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"训练集: {X_train.shape[0]}")
print(f"测试集: {X_test.shape[0]}")

# === 模型对比 ===
models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Linear SVM': LinearSVC(random_state=42),
}

results = {}
for name, model in models.items():
    # 训练
    model.fit(X_train, y_train)
    # 预测
    y_pred = model.predict(X_test)
    # 评估
    acc = accuracy_score(y_test, y_pred)
    results[name] = {'model': model, 'accuracy': acc}
    print(f"\n{name}:")
    print(f"  Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, 
                                target_names=['消极', '积极']))

# 选择最佳模型
best_model_name = max(results, key=lambda k: results[k]['accuracy'])
best_model = results[best_model_name]['model']
print(f"\n最佳模型: {best_model_name} (Accuracy={results[best_model_name]['accuracy']:.4f})")
```

### Step 5: 推理函数

```python
# step5_inference.py — 情感分析推理

class SentimentAnalyzer:
    """完整的情感分析器"""
    
    def __init__(self, preprocessor, vectorizer, model):
        self.preprocessor = preprocessor
        self.vectorizer = vectorizer
        self.model = model
    
    def predict(self, text):
        """预测单条文本的情感"""
        # 预处理
        processed = self.preprocessor.preprocess(text)
        # 向量化
        features = self.vectorizer.transform([processed])
        # 预测
        prediction = self.model.predict(features)[0]
        # 概率 (如果模型支持)
        try:
            proba = self.model.predict_proba(features)[0]
            confidence = max(proba)
        except:
            confidence = None
        
        sentiment = "积极" if prediction == 1 else "消极"
        
        return {
            'text': text,
            'sentiment': sentiment,
            'label': prediction,
            'confidence': confidence,
        }
    
    def batch_predict(self, texts):
        """批量预测"""
        return [self.predict(text) for text in texts]

# 创建分析器
analyzer = SentimentAnalyzer(preprocessor, vectorizer, best_model)

# === 测试 ===
test_texts = [
    "This is the best movie I have ever watched!",
    "Absolutely horrible. I want my money back.",
    "It was okay, nothing special but not terrible either.",
    "The cinematography was breathtaking and the music was perfect.",
    "Boring, slow, and completely predictable.",
]

print(f"\n{'='*60}")
print(f"情感分析测试")
print(f"{'='*60}")
for text in test_texts:
    result = analyzer.predict(text)
    print(f"\n文本: {result['text']}")
    print(f"情感: {result['sentiment']}", end='')
    if result['confidence']:
        print(f" (置信度: {result['confidence']:.2%})")
    else:
        print()
```

### Step 6: 可视化

```python
# step6_visualization.py — 结果可视化

import matplotlib.pyplot as plt
import numpy as np

# === 1. 模型对比柱状图 ===
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 模型准确率对比
model_names = list(results.keys())
accuracies = [results[n]['accuracy'] for n in model_names]
colors = ['#2196F3', '#4CAF50', '#FF9800']

axes[0].bar(model_names, accuracies, color=colors)
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Model Comparison')
axes[0].set_ylim(0, 1.1)
for i, v in enumerate(accuracies):
    axes[0].text(i, v + 0.02, f'{v:.2%}', ha='center', fontweight='bold')

# === 2. 词频分析 ===
# 积极评论高频词
pos_texts = df[df['label']==1]['processed_text']
pos_words = ' '.join(pos_texts).split()
pos_freq = pd.Series(pos_words).value_counts().head(10)

# 消极评论高频词
neg_texts = df[df['label']==0]['processed_text']
neg_words = ' '.join(neg_texts).split()
neg_freq = pd.Series(neg_words).value_counts().head(10)

axes[1].barh(pos_freq.index, pos_freq.values, color='#4CAF50', 
             alpha=0.7, label='积极')
axes[1].barh(neg_freq.index, -neg_freq.values, color='#F44336', 
             alpha=0.7, label='消极')
axes[1].set_xlabel('Frequency')
axes[1].set_title('Top Words by Sentiment')
axes[1].legend()

plt.tight_layout()
plt.savefig('sentiment_analysis_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("图表已保存为 sentiment_analysis_results.png")
```

### Step 7: 保存和加载

```python
# step7_save.py — 模型持久化

import joblib

# 保存完整管道
pipeline = {
    'preprocessor': preprocessor,
    'vectorizer': vectorizer,
    'model': best_model,
    'metadata': {
        'model_type': best_model_name,
        'accuracy': results[best_model_name]['accuracy'],
        'feature_count': X.shape[1],
    }
}

joblib.dump(pipeline, 'sentiment_analyzer.joblib')
print("模型已保存为 sentiment_analyzer.joblib")

# 加载
loaded = joblib.load('sentiment_analyzer.joblib')
loaded_analyzer = SentimentAnalyzer(
    loaded['preprocessor'],
    loaded['vectorizer'],
    loaded['model']
)

# 验证加载的模型
result = loaded_analyzer.predict("This movie is great!")
print(f"加载后预测: {result['sentiment']}")
```

## 完整 Pipeline 一键版

```python
# complete_pipeline.py — 一键运行完整流程

"""
完整的 NLP 情感分析 Pipeline
从数据加载到模型保存的端到端流程
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib

# Step 1: 数据
def load_data():
    """加载数据"""
    data = {
        'text': [
            "This movie is absolutely fantastic!",
            "Amazing storyline and brilliant acting.",
            "One of the best movies of the year.",
            "Stunning visuals and great soundtrack.",
            "Wonderful performance by the entire cast.",
            "Terrible plot, awful acting. Waste of time.",
            "Boring and predictable. Worst movie ever.",
            "The story makes no sense.",
            "Disappointing sequel. Nothing like original.",
            "Poor direction and bad effects.",
        ],
        'label': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    }
    return pd.DataFrame(data)

# Step 2: 预处理
def preprocess_text(text):
    """简单预处理"""
    import re
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# Step 3: 主流程
def main():
    # 数据
    df = load_data()
    df['text'] = df['text'].apply(preprocess_text)
    
    # 特征提取
    vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df['text'])
    y = df['label'].values
    
    # 划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # 训练
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    # 评估
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, 
                                target_names=['消极', '积极']))
    
    # 保存
    joblib.dump({'vectorizer': vectorizer, 'model': model}, 
                'model.joblib')
    print("✅ 完成! 模型已保存。")

if __name__ == '__main__':
    main()
```

## 关键概念回顾

```
本程序涉及的 NLP 概念:

┌──────────────────────────────────────────────────┐
│                                                  │
│  1. 文本预处理                                   │
│     ├── 清洗: 去HTML/URL/特殊字符                │
│     ├── 分词: 将文本切分为词                     │
│     ├── 停用词: 去掉无信息量的词(the, is, a)    │
│     └── 词形还原: running → run                 │
│                                                  │
│  2. 特征提取                                     │
│     └── TF-IDF: 词的重要性 = 频率 × 稀有度       │
│                                                  │
│  3. 分类模型                                     │
│     ├── 朴素贝叶斯: 基于概率                     │
│     ├── 逻辑回归: 线性分类器                     │
│     └── SVM: 最大间隔分类                        │
│                                                  │
│  4. 评估指标                                     │
│     ├── Accuracy: 准确率                         │
│     ├── Precision: 精确率                        │
│     ├── Recall: 召回率                           │
│     └── F1: 调和平均                              │
│                                                  │
└──────────────────────────────────────────────────┘
```

## 小结

| 步骤 | 技术 | 关键点 |
|------|------|--------|
| 数据准备 | pandas | 格式化为统一结构 |
| 预处理 | NLTK | 清洗→分词→去停用词→还原 |
| 特征提取 | TF-IDF | 文本→数值向量 |
| 训练 | sklearn | 对比多个模型选最优 |
| 评估 | classification_report | 多维度评估 |
| 部署 | joblib | 保存/加载模型 |

> 这就是 NLP 的完整流程。后续章节会深入每一步的原理和优化方法。

---

| [← 回到目录](../README.md) | [上一篇：Python环境搭建](03-Python环境搭建.md) | [下一篇：语音学与词法学](../02-语言学基础篇/01-语音学与词法学.md) |
|---|---|---|
