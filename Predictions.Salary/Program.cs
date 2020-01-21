using Microsoft.ML;
using Predictions.Salary.Models;
using System.Collections.Generic;

namespace Predictions.Salary
{
    class Program
    {
        internal static IList<Employee> Data { get; } = new List<Employee>();
        internal static ITransformer TrainedModel { get; set; }

        private static void Main(string[] args)
        {
            MainMenu.Show();
        }
    }
}
