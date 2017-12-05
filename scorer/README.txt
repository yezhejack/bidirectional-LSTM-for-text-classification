******************************************************
* SemEval-2016 Task 4: Sentiment Analysis on Twitter *
*                                                    *
*               SCORER                               *
*                                                    *
* http://alt.qcri.org/semeval2016/task4/             *
* semevaltweet@googlegroups.com                      *
*                                                    *
******************************************************


Version 2.3; February 24, 2016


Task organizers:

* Preslav Nakov, Qatar Computing Research Institute, HBKU
* Alan Ritter, The Ohio State University
* Sara Rosenthal, Columbia University
* Fabrizio Sebastiani, Qatar Computing Research Institute, HBKU
* Veselin Stoyanov, Facebook


*** NEW in version 2.3 ***

Fixed another bug with scorer for subtask B; it was switching P and R.


*** NEW in version 2.2 ***

Fixed a bug with the scorer for subtask B; it was calculating macro-averaged F1 instead of macro-averaged R. Also, fixed a bug in the scorer for subtask E.


*** NEW in version 2.1 ***

Fixed an issue with the aggregation scripts and with the format checker for subtask B.


*** NEW in version 2.0 ***

Added instructions about how to use the scorer as a format checker.


USE

To run the scorer (format checker) for task X (where X in [A..E]) run:
> perl score-semeval2016-task4-subtask{X}.pl GOLD_STANDARD_FILE PREDICTION_FILE

** For format checking, use the unlabeled test data as the GOLD_STANDARD_FILE. The scorer will output "Format OK" if the format checking passes. **

DATA FORMAT for gold standard and prediction files


-----------------------SUBTASK A-----------------------------------------

	id<TAB>label

where "label" can be 'positive', 'neutral' or 'negative'.


-----------------------SUBTASK B--------------------------------------

	id<TAB>topic<TAB>label

where "label" can be 'positive' or 'negative' (note: no 'neutral'!).

-----------------------SUBTASK C--------------------------------------

	id<TAB>topic<TAB>label

where "label" can be -2, -1, 0, 1, or 2,

-----------------------SUBTASK D--------------------------------------

	topic<TAB>positive<TAB>negative

where positive and negative are floating point numbers between 0.0 and 1.0 and positive + negative must sum to 1.0

-----------------------SUBTASK E--------------------------------------

	topic<TAB>label-2<TAB>label-1<TAB>label0<TAB>label1<TAB>label2

where label-2 to label2 are floating point numbers between 0.0 and 1.0 and the five numbers must sum to 1.0. label-2 corresponds to the fraction of tweets labeled -2 in the data and so on.


AGGREGATORS:
We also provide two aggregators that can turn tweet-level predictions into aggregate predictions needed for tasks D and E. Those can be used as the default aggregators. To run them run:
> perl aggregate-semeval2016-task4-subtask{X}.pl FILE
The aggregator will produce file FILE.aggregate with the aggregate predictions in it.

LICENSE

The accompanying dataset is released under a Creative Commons Attribution 3.0 Unported License (http://creativecommons.org/licenses/by/3.0/).



CITATION

You can cite the following paper when referring to the dataset:

@InProceedings{SemEval:2016:task4,
  author    = {Preslav Nakov and Alan Ritter and Sara Rosenthal and Veselin Stoyanov and Fabrizio Sebastiani},
  title     = {{SemEval}-2016 Task 4: Sentiment Analysis in {T}witter},
  booktitle = {Proceedings of the 10th International Workshop on Semantic Evaluation (SemEval 2016)},
  year      = {2016},
  publisher = {Association for Computational Linguistics}
}


USEFUL LINKS:

Google group: semevaltweet@googlegroups.com
SemEval-2016 Task 4 website: http://alt.qcri.org/semeval2016/task4/
SemEval-2016 website: http://alt.qcri.org/semeval2016/
