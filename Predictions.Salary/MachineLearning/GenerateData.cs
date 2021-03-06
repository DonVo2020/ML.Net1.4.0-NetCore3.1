﻿using Predictions.Salary.Infrastructure;
using Predictions.Salary.Models;
using Predictions.Salary.Services;
using System;
using System.Linq;

namespace Predictions.Salary.MachineLearning
{
    public static class GenerateData
    {
        private static readonly Random Random = new Random();

        public static void Execute()
        {
            Print.Header("Generate Data");

            Program.Data.Clear();

            const int numberOfEmployeesToGenerate = 100_000;
            //var numberOfEmployeesToGenerate = ConsoleHelper.GetNumber("How many employees do you want to generate?", 1_000_000);

            Console.WriteLine();

            ConsoleSpinner.Execute($"Generating {numberOfEmployeesToGenerate:N0} Employees", () =>
            {
                for (var i = 0; i < numberOfEmployeesToGenerate; i++)
                {
                    var age = Random.Next(20, 60);
                    var experienceLevel = ExperienceLevel.Values[Random.Next(0, 3)];
                    var employee = new Employee(age, experienceLevel);

                    Program.Data.Add(employee);
                }
            });

            Print.PreviewGeneratedData(Program.Data.Take(10));
        }
    }
}
