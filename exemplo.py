import nltk
#nltk.download('averaged_perceptron_tagger')
#nltk.download('maxent_ne_chunker')
#nltk.download('words')


texto = "Mr. Green killed Colonel Musterd in the study with clandlestick. Mr. Green is not very nice fellow"

#print(texto.split('.'))

frases = nltk.tokenize.sent_tokenize(texto)

#print(frases)

tokens = nltk.word_tokenize(texto)
#print(tokens)

#Categorização em classes dos tokens, exmeplo: Nome proprio, Verbo, verbo no passado, adverbio, adjetivo e etc
classes = nltk.pos_tag(tokens)
#print(classes)


entidades = nltk.chunk.ne_chunk(classes)
print(entidades)