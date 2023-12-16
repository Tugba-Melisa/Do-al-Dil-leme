# Doğal Dil İşleme (NLP) Nedir?
Doğal Dil İşleme (NLP), insan diliyle bilgisayarların nasıl etkileşim kurabileceğini inceleyen bir bilim dalıdır. NLP'nin temel amacı, insanların günlük kullandığı dili, bilgisayar programlarının anlayabileceği ve işleyebileceği bir formata dönüştürmektir.Bu alanda birçok temel kavram ve işlem bulunmaktadır:

* **Tokenization (Belirteçleme):** Metni küçük parçalara, yani belirteçlere bölmek. Bu belirteçler genellikle kelimeler veya cümleler olabilir.
* **Part-of-Speech Tagging (Kelime Türü Etiketleme):** Her kelimenin dil bilgisindeki rolünü belirleme, yani öznenin, nesnenin, fiilin vb. ne olduğunu etiketleme.
* **Named Entity Recognition (Adlandırılmış Varlık Tanıma):** Metindeki önemli isimleri (örneğin, kişilerin adları, organizasyonlar, yerler) tanıma.
* **Sentiment Analysis (Duygu Analizi):** Bir metnin içerdiği duygusal tonu belirleme, yani metnin olumlu, olumsuz veya nötr olduğunu anlama.
* **Dependency Parsing (Bağımlılık Ayrıştırma):** Cümledeki kelimeler arasındaki bağımlılıkları belirleme, yani hangi kelimenin diğerine bağlı olduğunu anlama.

# Makine Çevirisi (MT) Nedir?
Makine Çevirisi (MT), kaynak dilden hedef dile aktarımın otomatik yazılımlar aracılığıyla gerçekleştiği ve insan müdahalesinin olmadığı çeviri sürecidir. **Kural tabanlı makine çevirisi (RBMT)** ve **istatistiksel makine çevirisi (SMT)** olmak üzere farklı motorlar vardır. RBMT, dilbilgisi kurallarını kullanarak çeviri yaparken, SMT büyük veri setlerini analiz edip istatistiksel çeviri modellerini kullanır.
Son yıllarda, nöral ağlar kullanılarak yapılan **nöral makine çevirisi (NMT)** büyük ilgi görmektedir. NMT, desenler ve yapılar aracılığıyla metinleri çevirmek için derin öğrenmeyi kullanır. Bu yaklaşım, daha akıcı çeviriler sağlayarak diğer makine çevirisi yöntemlerine göre daha iyi performans gösterir. 
### Makine Çevirisi Kullanım Alanları?
* **Çeviri uygulamaları** : MT, farklı dilleri konuşan kişiler arasındaki dil bariyerlerini aşmada önemli bir araçtır. İki veya daha fazla dil konuşan kişiler, MT sayesinde metin veya konuşmalarını anadillerine çevirebilirler. Popüler çeviri uygulamaları şunlardır Google Translate, DeepL, Amazon Translate, Microsoft Bing, Systrian, Crowdin, Smartling
  
Code example for Extractive summarization with BERT

```python
# Encode the sentences using the SBERT model
sentence_embeddings = model.encode(sentences)

# Calculate the cosine similarity between each sentence embedding
similarity_matrix = np.zeros([len(sentences), len(sentences)])
for i in range(len(sentences)):
    for j in range(len(sentences)):
        if i != j:
            similarity_matrix[i][j] = util.cos_sim(sentence_embeddings[i], sentence_embeddings[j])
print(similarity_matrix)
sentence_scores = np.sum(similarity_matrix, axis=1)

# Select the top 3 sentences with the highest scores
summary_sentences = []
for i in range(3):
    index = np.argmax(sentence_scores)
    summary_sentences.append(sentences[index].strip())
    sentence_scores[index] = -1

# Concatenate the summary sentences to create a summary
summary = '. '.join(summary_sentences)
print(summary)
doc = nlp(metin)

# Anahtar cümleleri ve bilgileri özetleme
ozet = " ".join([cümle.text for cümle in doc.sents][:2])  # İlk iki cümleyi alarak özetleme

# Sonucu yazdırın
print(ozet)
```
1^ Assume that we separete text for extract sentences.
Code Example for Abstractive summarization with Huggingface

kaynak düzenlencek(https://christianbernecker.medium.com/nlp-summarization-use-bert-and-bart-to-summarize-your-favourite-newspaper-articles-fb9a81bed016)
```python
# install Transformers: pip install transformers
from transformers import pipeline
# This dataset is powered by Facebook and has been created to utilize the large dataset of CNN and Daily Mail.
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

ARTICLE = """ """
print(summarizer(text, max_length=130, min_length=30, do_sample=False))
```
### Okey but what is transformers?
The original Transformer is based on an encoder-decoder architecture and is a classic sequence-to-sequence model.
There are many ways to solve a given task, some models may implement certain techniques or even approach the task from a new angle, but for Transformer models, the general idea is the same. Owing to its flexible architecture, most models are a variant of an encoder, decoder, or encoder-decoder structure. 
BERT was first designed for machine translation but has become a common model for various language tasks. In text classification, BERT is an encoder-only model, which means it focuses on understanding words from both directions. To use BERT for text classification, a sequence classification head is added. It's a linear layer converting final hidden states to logits for predicting labels. Cross-entropy loss is then calculated to find the most likely label.


**Attention Mechanism:** The fundamental feature of the Transformer is an attention mechanism that enables each element in the input sequence to focus on other elements. This allows each output element, especially words or tokens, to assign weights to all other elements in the input.

**Encoder-Decoder Architecture**: The Transformer typically consists of two main parts: an encoder and a decoder. The encoder encodes input data in a way that can be better represented. The decoder then transforms this representation into output data.

**Multi-Head Attention:** The attention mechanism is expanded using multiple heads. Each head can focus on different features, allowing the model to learn more complex relationships.


**Layer Normalization and Feedforward Networks:** Each encoder and decoder layer includes layer normalization and a feedforward network. Layer normalization stabilizes training by normalizing the inputs to each layer. The feedforward network helps transform the representations of tokens at each position in a more complex way.
https://huggingface.co/docs/transformers/tasks_explained
# Natural Language Processing for Finance
NLP has many applications and benefits in finance, such as the ability to analyze
sentiment and extract information from financial text data, to automate the generation of trading
signals and risk alerts, and to inform decision-making and risk management in the finance
industry.
kaynak : A Primer on Natural Language Processing for Finance Prof. Dr. Joerg Osterrieder
## Some examples of how NLP is used in finance:
* **Classifying Financial Documents**
  
    It is capable of automated classification across various **agreement types**, including loans, service agreements, and consulting agreements. It excels in categorizing news based on **ESG** (Environmental, Social, and Governance) criteria. Additionally, it efficiently identifies topics within **banking-related texts** and classifies tickets based on their topic or class. Furthermore, our system can accurately determine **sentiment**, distinguishing between negative, positive, and neutral expressions.
* **Recognizing Financial Entities**
  
    NLP helps us identify and classify named entities in text, such as **people, locations, dates, numbers, etc**. to make recommendations or predictions.
* Understanding Entities in Context
  
    Understanding entities in context is the ability of asserting if an entity is mentioned to happen in the present, past, future, if it’s negated, present, absent, if it’s hypothetical, probable, etc.
* **Extracting Financial Relationships**
  
    Relation Extraction is the ability to infer if two entities are connected. It helps us extract relations between a company and its profit, losses, cash flow operations, etc.
* Normalization and Data Augmentation
  
    **Normalization** makes words more consistent and helps the model work better. When we normalize text, we make it less random and align it with a set standard.
    **Data Augmentation** refers to the capability of using extracted information, such as Company Names, to inquire from data sources and acquire additional information. This can include details like the Company's SIC code, Trading Symbol, Address, Financial Period, and more


https://www.johnsnowlabs.com/examining-the-impact-of-nlp-in-financial-services/
<div style="text-align: center;">
  <p>Firstly, let's look at the nlp investments made in the financial sector</p>
  <img src="https://github.com/mehmetismostlyclear/NlpFinansRaporu/assets/87391205/7c7483d3-de7d-47cc-a802-4c6fa12f9b8b" alt="Resim" width="500" />
</div>
<div style="text-align: center;">
  <p>Firstly, let's look at the nlp investments made in the financial sector</p>
  <img src="https://github.com/mehmetismostlyclear/NlpFinansRaporu/assets/87391205/0fedcfd2-6145-479e-b6a7-5606ef2ec458" alt="Resim" width="500" />
</div>



kaynak: www.towardsdatascience.com/nlp-startup-funding-in-2022
