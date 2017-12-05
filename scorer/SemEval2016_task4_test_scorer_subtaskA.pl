#!/usr/bin/perl
#
#  Author: Preslav Nakov
#  
#  Description: Scores SemEval-2016 task 4, subtask A
#               Calculates macro-average R, macro-average F1, and Accuracy
#
#  Last modified: February 25, 2016
#
#

use warnings;
use strict;
use utf8;
binmode (STDIN,  ":utf8");
binmode (STDOUT, ":utf8");

my $INPUT_FILE         =  $ARGV[0];
use constant GOLD_FILE => 'SemEval2016_task4_subtaskA_test_gold.txt';
my $OUTPUT_FILE        =  $INPUT_FILE . '.scored';


########################
###   MAIN PROGRAM   ###
########################

my %stats = ();

### 1. Read the files and get the statsitics
open INPUT, $INPUT_FILE or die;
open GOLD,  GOLD_FILE or die;
#open INPUT, '<:encoding(UTF-8)', $INPUT_FILE or die;
#open GOLD,  '<:encoding(UTF-8)', GOLD_FILE or die;

my $lineNo = 1;
for (; <INPUT>; $lineNo++) {
	s/^[ \t]+//;
	s/[ \t\n\r]+$//;

	### 1.1. Check the input file format
	#1	positive	i'm done writing code for the week! Looks like we've developed a good a** game for the show Revenge on ABC Sunday, Premeres 09/30/12 9pm
	die "Wrong file format for $INPUT_FILE: $_" if (!/^$lineNo\t(positive|negative|neutral)/);
	my $proposedLabel = $1;

	### 1.2	. Check the gold file format
	#NA	T14114531	positive
	$_ = <GOLD>;
	die "Wrong file format!" if (!/^$lineNo\t([^\t]+)\t(positive|negative|neutral)/);
	my ($dataset, $trueLabel) = ($1, $2);

	### 1.3. Update the statistics
	$stats{$dataset}{$proposedLabel}{$trueLabel}++;
}

close(INPUT) or die;
close(GOLD) or die;
$lineNo--;
die "Too few lines: $lineNo" if (32009 != $lineNo);

### 2. Initialize zero counts
foreach my $dataset (keys %stats) {
	foreach my $class1 ('positive', 'negative', 'neutral') {
		foreach my $class2 ('positive', 'negative', 'neutral') {
			$stats{$dataset}{$class1}{$class2} = 0 if (!defined($stats{$dataset}{$class1}{$class2}))
		}
	}
}

### 3. Calculate the F1 for each dataset
print "$INPUT_FILE\t";
open OUTPUT, '>:encoding(UTF-8)', $OUTPUT_FILE or die;
foreach my $dataset (sort keys %stats) {
	print OUTPUT "\nScoring $dataset:\n";
	print "$dataset\t";

	my $avgR  = 0.0;
	my $avgF1 = 0.0;
	foreach my $class ('positive', 'negative', 'neutral') {

		my $denomP = (($stats{$dataset}{$class}{'positive'} + $stats{$dataset}{$class}{'negative'} + $stats{$dataset}{$class}{'neutral'}) > 0) ? ($stats{$dataset}{$class}{'positive'} + $stats{$dataset}{$class}{'negative'} + $stats{$dataset}{$class}{'neutral'}) : 1;
		my $P = $stats{$dataset}{$class}{$class} / $denomP;

		my $denomR = ($stats{$dataset}{'positive'}{$class} + $stats{$dataset}{'negative'}{$class} + $stats{$dataset}{'neutral'}{$class}) > 0 ? ($stats{$dataset}{'positive'}{$class} + $stats{$dataset}{'negative'}{$class} + $stats{$dataset}{'neutral'}{$class}) : 1;
		my $R = $stats{$dataset}{$class}{$class} / $denomR;
				
		my $denom = ($P+$R > 0) ? ($P+$R) : 1;
		my $F1 = 2*$P*$R / $denom;

	    $avgR  += $R;
		$avgF1 += $F1 if ($class ne 'neutral');
		printf OUTPUT "\t%8s: P=%0.4f, R=%0.4f, F1=%0.4f\n", $class, $P, $R, $F1;
	}

	$avgR  /= 3.0;
	$avgF1 /= 2.0;
	my $acc = ($stats{$dataset}{'positive'}{'positive'} + $stats{$dataset}{'negative'}{'negative'} + $stats{$dataset}{'neutral'}{'neutral'}) 
	        / ($stats{$dataset}{'positive'}{'positive'} + $stats{$dataset}{'negative'}{'negative'} + $stats{$dataset}{'neutral'}{'neutral'} +
	           $stats{$dataset}{'positive'}{'negative'} + $stats{$dataset}{'negative'}{'positive'} +
	           $stats{$dataset}{'positive'}{'neutral'} + $stats{$dataset}{'neutral'}{'positive'} +
	           $stats{$dataset}{'negative'}{'neutral'} + $stats{$dataset}{'neutral'}{'negative'});
	printf OUTPUT "\t\tAvgF1_2=%0.3f, AvgR_3=%0.3f, Acc=%0.3f\n", $avgF1, $avgR, $acc;
	printf OUTPUT "\tOVERALL SCORE : %0.3f\n", $avgF1;
	printf "%0.3f\t%0.3f\t%0.3f\t", $avgF1, $avgR, $acc;

}
print "\n";
close(OUTPUT) or die;
