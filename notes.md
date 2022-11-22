36 random moves => ~ 2s train time
35 random moves => ~ 7s train time
34 random moves => ~14s train time
33 random moves => ~40s train time

Via exponential regression, the train time seems to follow a shape of:
y = 3004643536953960.5 ⋅ 0.3798^x

so estimate
10 random moves => ~187647182798s = 54 million hours
20 random moves => ~11719029s = 3255 hours
28 random moves => ~5073s = 82m = 1.1 hours
30 random moves => ~731s = 12m = 0.2 hours

for the final one maybe train on 28 moves?
theory: we end up getting more meaningful datapoints from the longer training session - so even though it takes longer we get more data (and it is more meaningful)


for the tester, lets do maybe 100 games with 34 random moves






### How to organize the dataset

train:
  board states
  2x7x7 - our stones, opponents stones

label:
  the played move
  1x49 - one hot encoded




### Template for end solver

1: use heuristics for first n = ~10 Moves
2: swap to neural net predictor for rest of Moves
  - nn will always predict some legal move (because we remove illegal predictions and just normalize)
3: if prediction takes > 29s, play random legal move
