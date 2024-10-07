#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

int CountRowMismatches(vector<string> first, vector<string> second);
int CountColumnMismatches(vector<string> first, vector<string> second);


int main()
{
	ifstream inputFile;
	string inputFilePath = "F:/Canada/Concordia/COEN6321_machineLearning/Assignment#1/Assignment1CPP/Ass1Output.txt";
	inputFile.open(inputFilePath);

	string currentText;
	getline(inputFile, currentText);

	vector<vector<string>> puzzlePieces;
	string currentNumber;

	const int RowSize = 8;
	const int ColumnSize = 8;
	const char delimiter = ' ';

	for (int i = 0; i < RowSize; i++)
	{
		vector<string> row;

		getline(inputFile, currentText);

		stringstream currentLine(currentText); 

		while (getline(currentLine, currentNumber, delimiter))
			row.push_back(currentNumber);

		puzzlePieces.push_back(row);
	}

	int numberOfMismatches = 0;

	for (int i = 0; i < RowSize - 1; i++)
		numberOfMismatches += CountRowMismatches(puzzlePieces[i], puzzlePieces[i + 1]);

	for (int i = 0; i < ColumnSize - 1; i++)
	{
		vector<string> firstColumn;
		vector<string> secondColumn;

		for (int j = 0; j < RowSize; j++)
		{
			firstColumn.push_back(puzzlePieces[j][i]);
			secondColumn.push_back(puzzlePieces[j][i + 1]);
		}

		numberOfMismatches += CountColumnMismatches(firstColumn, secondColumn);
	}

	cout << "Number of mismatches : " << numberOfMismatches << endl;

	inputFile.close();
}

int CountRowMismatches(vector<string> first, vector<string> second)
{
	int numberOfMismatches = 0;

	for (int i = 0; i < first.size(); i++)
		if (first[i].at(2) != second[i].at(0))
			numberOfMismatches++;

	return numberOfMismatches;
}

int CountColumnMismatches(vector<string> first, vector<string> second)
{
	int numberOfMismatches = 0;

	for (int i = 0; i < first.size(); i++)
		if (first[i].at(1) != second[i].at(3))
			numberOfMismatches++;

	return numberOfMismatches;
}