# sentiment

## Word Embeddings
To create word embeddings, we tokenized our input data and applied Glove. For tokenizing, we used the Stanford Core NLP package. First we combined all the unlabeled reviews with data from the train set into one document. Then we applied the Stanford tokenizer:

```
java -cp stanford-corenlp-3.9.1.jar:stanford-corenlp-3.9.1-models.jar edu.stanford.nlp.process.PTBTokenizer -preserveLines -lowerCase input_data.txt --add-modules java.se.ee >tokenized_data.txt
```

Then we used the demo.sh file from Glove to create our word embeddings. For this, we used Glove's default parameters and only changed the dimensions of the vectors to 25 and 200. As a corpus, we used our tokenized data.

## Training models
To train the models, a tokenized version of the data and the word embeddings is needed. Instructions to download these files can be found in the folder **Data** and **Embeddings**.
