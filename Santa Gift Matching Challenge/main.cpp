
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define true 1
#define false 0

const unsigned int N_CHILDREN = 1000000;
const unsigned int N_GIFT_PREF = 10;
const unsigned int N_GIFT_TYPE = 1000;
const unsigned int N_CHILD_PREF = 1000;
const unsigned int N_TWINS = 4000;

// 19711 is the maximum number of children that wish the same gift
const unsigned int N_CANDIDATES_PER_GIFT = 1000 + 19711;
const unsigned int N_ROUNDS = 3;

int **parseCsvFile(const char *path, int nRows, int nColumns, bool hasHeader, bool skipFirstColumn);

int *parseSolution(const char *path);

int **getCandidatesMatrix(const int **giftPref, const int **childPref);

void saveToCsv(const int *state, const char *path);

int **getScoreMatrix(const int **giftPref, const int **childPref);

long long getScore(const int *solution, const int **scoreMatrix);

int main() {

    const int **const giftPref = (const int **const) parseCsvFile("/home/davinci/kaggle_santa/child_wishlist.csv", N_CHILDREN,
                                                                  N_GIFT_PREF + 1, false, true);
    const int **const childPref = (const int **const) parseCsvFile("/home/davinci/kaggle_santa/gift_goodkids.csv", N_GIFT_TYPE,
                                                                   N_CHILD_PREF + 1, false, true);
    int *state = parseSolution("/home/davinci/kaggle_santa/submission.csv");

    const int **const scoreMatrix = (const int **const) getScoreMatrix(giftPref, childPref);
    const int **const candidatesMatrix = (const int **const) getCandidatesMatrix(giftPref, childPref);

    long long score = getScore(state, scoreMatrix);

    for (int round = 1; round <= N_ROUNDS; round++) {
        for(int child = 4000; child < N_CHILDREN; child++) {
            int childGift = state[child];
            for (int i = 0; i < N_CANDIDATES_PER_GIFT && candidatesMatrix[childGift][i] != -1; i++) {
                int candidate = candidatesMatrix[childGift][i];

                // Don't swap with twins
                if (candidate < N_TWINS) continue;

                int candidateGift = state[candidate];

                // Calculate the score change
                int scoreChange = 0;
                scoreChange -= scoreMatrix[childGift][child];
                scoreChange -= scoreMatrix[candidateGift][candidate];
                scoreChange += scoreMatrix[candidateGift][child];
                scoreChange += scoreMatrix[childGift][candidate];

                if (scoreChange > 0) {
                    // Swap it!
                    state[candidate] = childGift;
                    state[child] = candidateGift;
                    score += scoreChange;

                    // Try the next child
                    break;
                }
            }

            if (child % 100000 == 0) {
                printf("Round: %d, Child: %d, Score: %lld \\n", round, child, score);
            }
        }

        // Print the average normalized happiness
        score = getScore(state, scoreMatrix);
        float anh = ((float) score) / 2000000000;
        printf("Round: %d finished, Average Normalized Happiness %.9g\\n", round, anh);

        // Save to csv
        printf("Saved to CSV file\\n");
        saveToCsv(state, "c_sub.csv");
    }
}

int **getCandidatesMatrix(const int **giftPref, const int **childPref) {
    // Init the candidate matrix with zeros
    int **candidateMatrix = static_cast<int**>(malloc(N_GIFT_TYPE * sizeof *candidateMatrix));
    int *lastCandidatePositions = static_cast<int*>(malloc(N_GIFT_TYPE * sizeof lastCandidatePositions));
    for (int gift = 0; gift < N_GIFT_TYPE; gift++) {
        candidateMatrix[gift] = static_cast<int*>(calloc(N_CANDIDATES_PER_GIFT, sizeof *candidateMatrix[gift]));
        lastCandidatePositions[gift] = -1;
    }

    // Add all good children for each gift
    for (int gift = 0; gift < N_GIFT_TYPE; gift++) {

        // Add all good children
        for (int iGoodChild = 0; iGoodChild < N_CHILD_PREF; iGoodChild++) {
            int child = childPref[gift][iGoodChild];
            int insertPos = lastCandidatePositions[gift] + 1;

            if (insertPos < N_CANDIDATES_PER_GIFT) {
                candidateMatrix[gift][insertPos] = child;
                lastCandidatePositions[gift] = insertPos;
            }
        }
    }

    // Add each child to each gift on its wishlist
    for (int child = 0; child < N_CHILDREN; child++) {
        for (int wish = 0; wish < N_GIFT_PREF; wish++) {
            int gift = giftPref[child][wish];
            int insertPos = lastCandidatePositions[gift] + 1;

            if (insertPos < N_CANDIDATES_PER_GIFT) {
                candidateMatrix[gift][insertPos] = child;
                lastCandidatePositions[gift] = insertPos;
            }
        }
    }

    return candidateMatrix;
}

long long getScore(const int *solution, const int **scoreMatrix) {
    long long score = 0;
    for (int child = 0; child < N_CHILDREN; child++) {
        int gift = solution[child];
        score += scoreMatrix[gift][child];
    }
    return score;
}

int **getScoreMatrix(const int **giftPref, const int **childPref) {
    // Init the score matrix with -101
    int **scoreMatrix = static_cast<int**>(malloc(N_GIFT_TYPE * sizeof *scoreMatrix));
    for (int gift = 0; gift < N_GIFT_TYPE; gift++) {
        scoreMatrix[gift] = static_cast<int*>(malloc(N_CHILDREN * sizeof *scoreMatrix[gift]));
        for (int child = 0; child < N_CHILDREN; child++) {
            scoreMatrix[gift][child] = -101;
        }
    }

    // Calculate the scores for the goodkids, i.e. add the gift happiness
    for (int gift = 0; gift < N_GIFT_TYPE; gift++) {
        for (int iGoodKid = 0; iGoodKid < N_CHILD_PREF; iGoodKid++) {
            int child = childPref[gift][iGoodKid];
            scoreMatrix[gift][child] += (N_CHILD_PREF - iGoodKid) * 2 + 1;
        }
    }

    // Calculate the scores for the wishlists, i.e. add the child happiness
    for (int child = 0; child < N_CHILDREN; child++) {
        for (int iWish = 0; iWish < N_GIFT_PREF; iWish++) {
            int gift = giftPref[child][iWish];
            scoreMatrix[gift][child] += (N_GIFT_PREF - iWish) * 200 + 100;
        }
    }
    return scoreMatrix;
}

int *parseSolution(const char *path) {
    int **const parsedCsv = parseCsvFile(path, N_CHILDREN, 2, true, true);
    // parsedCsv will have N_CHILDREN rows and 1 column
    int *solution = static_cast<int*>(malloc(N_CHILDREN * sizeof solution));
    for (int i = 0; i < N_CHILDREN; i++) {
        solution[i] = parsedCsv[i][0];
    }
    return solution;
}

void saveToCsv(const int *state, const char *path) {
    FILE *f = fopen(path, "w");
    if (f == NULL) {
        fprintf(stderr, "Could not open csv file %s\\n", path);
        return;
    }

    fprintf(f, "ChildId,GiftId\\n");
    for (int child = 0; child < N_CHILDREN; child++) {
        fprintf(f, "%d,%d\\n", child, state[child]);
    }
    fclose(f);
}

int **parseCsvFile(const char *path, int nRows, int nColumns, bool hasHeader, bool skipFirstColumn) {
    FILE *stream = fopen(path, "r");
    int resultColumns = skipFirstColumn ? nColumns - 1 : nColumns;
    int **result = static_cast<int**>(malloc(sizeof *result * nRows));
    char line[10000]; // Maximum line length: 10 kb

    // Skip the header
    if (hasHeader) fgets(line, 10000, stream);

    for (int row = 0; row < nRows; row++) {
        // Init the row
        result[row] = static_cast<int*>(malloc(sizeof *result[row] * resultColumns));
        // Read a line
        if (fgets(line, 10000, stream)) {

            // Split the line into tokens using the delimiter
            char *token;
            for (int column = 0; column < nColumns; column++) {
                if (column == 0) {
                    token = strtok(line, ",");
                } else {
                    // Keep reading tokens
                    token = strtok(NULL, ",");
                }

                int resultColumn = skipFirstColumn ? column - 1 : column;
                if (token) {
                    if (resultColumn >= 0) result[row][resultColumn] = atoi(token);
                } else {
                    fprintf(stderr, "Could not read line %d, token %d from file %s\\n", row, column, path);
                }
            }
        } else {
            fprintf(stderr, "Could not read line %d from file %s\\n", row, path);
        }
    }
    return result;
}


