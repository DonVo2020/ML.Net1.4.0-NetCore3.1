﻿
#pragma warning disable 649 // We don't care about unsused fields here, because they are mapped with the input file.

using Microsoft.ML.Data;

namespace MulticlassClassificationsML.GitHubLabeler.DataStructures
{
    internal class GitHubIssuePrediction
    {
        [ColumnName("PredictedLabel")]
        public string Area;

        public float[] Score;
    }
}
