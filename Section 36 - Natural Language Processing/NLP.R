library(tm)
library(SnowballC)
datasetog= read.delim('Restaurant_Reviews.tsv', quote='', stringsAsFactors = FALSE)
corpus=VCorpus(VectorSource(datasetog$Review))
corpus= tm_map(corpus, content_transformer(tolower))
corpus= tm_map(corpus, removePunctuation)
corpus= tm_map(corpus, removeNumbers)
corpus= tm_map(corpus, removeWords, stopwords())
corpus= tm_map(corpus, stemDocument)
corpus= tm_map(corpus, stripWhitespace)

dtm= DocumentTermMatrix(corpus)
dtm= removeSparseTerms(dtm, 0.999)
dataset= as.data.frame(as.array(dtm))
dataset$Liked= datasetog$Liked
dataset$Liked= factor(dataset$Liked, levels= c(0,1))
library(caTools)
library(randomForest)
set.seed(123)
split= sample.split(dataset$Liked, SplitRatio = 0.8)
trainingset= subset(dataset, split==TRUE)
testset= subset(dataset, split==FALSE)

classifier= randomForest(x=trainingset[-692],
                         y=trainingset$Liked,
                         ntree= 10)
ypred= predict(classifier, testset[-692])
cm=table(testset[-692], ypred)




