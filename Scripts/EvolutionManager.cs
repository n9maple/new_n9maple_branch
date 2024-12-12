using UnityEngine;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using UnityEngine.Serialization;

#if UNITY_EDITOR
using UnityEditor;
#endif

internal class Genotype
{
    public readonly List<double> Weights;
    public float Fitness;

    public Genotype(List<double> weights, float fitness)
    {
        Weights = weights;
        Fitness = fitness;
    }
}

internal class Vehicle
{
    public Genotype Genotype;
    public GameObject GameObject;
    public int ID;
    public VehicleController Controller;
    public DateTime StartTime;
    public bool finished = false;
   

    public Vehicle(Genotype genotype, GameObject gameObject)
    {
        Genotype = genotype;
        GameObject = gameObject;
        ID = gameObject.GetInstanceID();
        Controller = gameObject.GetComponent<VehicleController>();
    }

    public void Reset()
    {
        finished = false;
        StartTime = DateTime.Now;
    }

    public float GetFitness()
    {
        return (float)(DateTime.Now - StartTime).TotalSeconds;
    }
}

public class EvolutionManager : MonoBehaviour
{
    
    public int populationSize = 15;
    public GameObject prefab;
    public float crossoverChance = 0.3f;
    public float mutationRate = 0.15f;
    public float mutationAmount = 1f;
    public int tournamentSize = 5;
    private bool stop = false;

    public List<int> layerSizes = new() { 5, 4, 1 };
    // activation functions used at each non-input layer (relu, sigmoid, tanh)
    public List<string> activationFuncList = new() { "tanh", "tanh" };
    
    private readonly Dictionary<int, Vehicle> _vehicles = new ();
    private readonly System.Random _rand = new ();

    private bool firstFinish = true;
    private int firstFinishGeneration;
    private int firstAllPassGeneration;
    private int generationNum = 0;

    private List<float> bestFitnessInGeneration = new();
    private List<float> averageFitnessInGeneration = new();
    private List<int> finishedVehiclesInGeneration = new();

    private List<Genotype> Genotypes
    {
        get
        {
            var result = new List<Genotype>();

            foreach (var (_, vehicle) in _vehicles)
                result.Add(vehicle.Genotype);

            return result;
        }

        set
        {
            var index = 0;
            
            foreach (var (_, vehicle) in _vehicles)
                vehicle.Genotype = value[index++];
        }
    }
    
    private void Awake()
    {
        Application.runInBackground = true;
    }

    private void Start()
    {
        // Disable collision between vehicles
        int vehicleLayer = LayerMask.NameToLayer("Vehicle");
        Physics.IgnoreLayerCollision(vehicleLayer, vehicleLayer, true);
        
        var genotypes = GenerateGenotypes();
        InstantiateVehicles(genotypes);
        SpawnVehicles();
    }

    private List<Genotype> GenerateGenotypes()
    {
        var genotypes = new List<Genotype>();
        
        for (var i = 0; i < populationSize; i++)
        {
            var weights = new List<double>();
            
            for (var j = 0; j < layerSizes.Count - 1; j++)
            {
                weights.AddRange(InitNegOneToOne(layerSizes[j], layerSizes[j + 1]));
                weights.AddRange(GenerateGaussianList(layerSizes[j + 1]));
            }

            genotypes.Add(new Genotype(weights, 0));
        }

        return genotypes;
    }

    private void InstantiateVehicles(List<Genotype> genotypes)
    {
        for (var i = 0; i < populationSize; i++)
        {
        
            var instance = Instantiate(prefab);
            var vehicle = new Vehicle(genotypes[i], instance);

            vehicle.Controller.OnFinish += () =>
            {
                markFinish(vehicle);
            };

            vehicle.Controller.OnHitWall += () =>
            {
                DisableVehicle(vehicle);
            };

            vehicle.Reset();
            _vehicles.Add(vehicle.ID, vehicle);
        }
    }
    
    // ReSharper disable Unity.PerformanceAnalysis
    private void SpawnVehicles()
    {
        // allWeights is the list of weights for all vehicles
        // for each vehicle, the weights list format is as follows
        // [layer1->2 weights. layer2 biases, layer2->3 weights, layer3 biases]
        // the list is flattened to 1D

        generationNum++;

        foreach ((int _, var vehicle) in _vehicles)
        {
            vehicle.Reset();
            
            var genotype = vehicle.Genotype;
            var instance = vehicle.GameObject;

            instance.transform.position = gameObject.transform.position;
            instance.transform.rotation = Quaternion.LookRotation(Vector3.forward);

            instance.SetActive(true);

            var weights = new List<List<double>>();
            var biasList = new List<List<double>>(); // biases that will be added to each non-input layer

            var lastIndex = 0;

            for (var j = 0 ; j < layerSizes.Count - 1; j++)
            {
                int weightsLen = layerSizes[j] * layerSizes[j + 1];
                weights.Add(genotype.Weights.GetRange(lastIndex, weightsLen));

                lastIndex += weightsLen;

                biasList.Add(genotype.Weights.GetRange(lastIndex, layerSizes[j + 1]));

                lastIndex += layerSizes[j + 1];
            }

            vehicle.Controller.Fnn = new NN(layerSizes, weights, biasList, activationFuncList);
        }
    }

    private void DisableVehicle(Vehicle vehicle)
    {
        vehicle.Genotype.Fitness = vehicle.GetFitness();
        vehicle.GameObject.SetActive(false);

        bool allDisabled = _vehicles.Values.All(v => !v.GameObject.activeSelf);
        

        if (allDisabled)
        {
            float sum_fitness = 0;
            foreach (var g in Genotypes)
                sum_fitness += g.Fitness;

            averageFitnessInGeneration.Add(sum_fitness / populationSize);

            float max_fit = 0;
            foreach (var g in Genotypes)
                if (g.Fitness > max_fit)
                    max_fit = g.Fitness;
            
            bestFitnessInGeneration.Add(max_fit);
            Debug.Log($"Best Fitness is ${max_fit}");

            int numFinished = _vehicles.Values.Count(v => v.finished);
            finishedVehiclesInGeneration.Add(numFinished);
            Debug.Log($"Generation {generationNum}. Finsihed vehicles: {numFinished}");
            RespawnVehicles();
        }
    }

    private void markFinish(Vehicle vehicle)
    {
        if (firstFinish)
        {
            firstFinish = false;
            firstFinishGeneration = generationNum;
            Debug.Log($"First lap at generation {firstFinishGeneration}\n");
        }

        vehicle.finished = true;

        bool Finished = _vehicles.Values.Count(v => v.finished) >= populationSize / 5;

        if (Finished)
        {

            stop = true;
        }
    }

    private void RespawnVehicles()
    {
        if (stop)
        {
            Debug.Log($"First lap: {firstFinishGeneration}\n20%: {generationNum}\nBest: {String.Join(", ", bestFitnessInGeneration)}\nAvg: {String.Join(", ", averageFitnessInGeneration)}\nPass: {String.Join(", ", finishedVehiclesInGeneration)}");
            EditorApplication.isPlaying = false;
        }
        // reproduction
        // var intermediateGeneration = RemainderStochasticSampling(Genotypes);
        var intermediateGeneration = Tournament(Genotypes);
        var newGeneration = Recombination(intermediateGeneration);
        
        Genotypes = Mutate(newGeneration);

        SpawnVehicles();
    }

    private List<Genotype> RemainderStochasticSampling(List<Genotype> genotypes)
    {
        var generation = new List<Genotype>();

        float fitnessSum = genotypes.Sum(g => g.Fitness);

        foreach (var g in genotypes)
        {
            float expectedCount = g.Fitness / fitnessSum * populationSize;

            var copies = (int)expectedCount;
            float extraCopyChance = expectedCount - copies;

            for (var i = 0; i < copies; i++)
                generation.Add(g);

            if (_rand.NextDouble() < extraCopyChance)
                generation.Add(g);
        }

        return generation;
    }

    private List<Genotype> Tournament(List<Genotype> genotypes)
    {
        var generation = new List<Genotype>();

        for (int i = 0; i < populationSize; i++)
        {
            // 隨機選出 tournamentSize 個基因型
            var tournament = new List<Genotype>();
            for (int j = 0; j < tournamentSize; j++)
            {
                var randomIndex = _rand.Next(genotypes.Count);
                tournament.Add(genotypes[randomIndex]);
            }
            // 選出適應度最高的基因型作為親代
            var bestGenotype = tournament.OrderByDescending(g => g.Fitness).First();
            // 加入到新一代中
            generation.Add(bestGenotype);
        }
        return generation;
    }

    private List<Genotype> Recombination(List<Genotype> intermediateGeneration)
    {
        var newGeneration = new List<Genotype>();
        intermediateGeneration.Sort((a, b) => b.Fitness.CompareTo(a.Fitness));

        //Debug.Log("Best Fitness in population: " + intermediateGeneration[0].Fitness);
        // keep best 2

        // only enable this in remainder stochastic
        newGeneration.Add(intermediateGeneration[0]);
        newGeneration.Add(intermediateGeneration[1]);

        while (newGeneration.Count < populationSize)
        {
            int ind1 = _rand.Next(0, intermediateGeneration.Count), ind2 = _rand.Next(0, intermediateGeneration.Count);
                
            while (ind1 == ind2)
                ind2 = _rand.Next(0, intermediateGeneration.Count);


            Genotype offspring1, offspring2;
            //(offspring1, offspring2) = CompleteCrossover(intermediateGeneration[ind1], intermediateGeneration[ind2]);
            (offspring1, offspring2) = OnepointCrossover(intermediateGeneration[ind1], intermediateGeneration[ind2]);

            newGeneration.Add(offspring1);

            if (newGeneration.Count < populationSize)
                newGeneration.Add(offspring2);
        }

        return newGeneration;
    }

    private (Genotype, Genotype) CompleteCrossover(Genotype parent1, Genotype parent2)
    {
        List<double> weights1 = new List<double>(),
                     weights2 = new List<double>();

        int weightsCount = parent1.Weights.Count;

        if (parent2.Weights.Count != weightsCount)
            Debug.LogError("Crossover parents have different sizes");

        for (var i = 0; i < weightsCount; i++)
        {
            if (_rand.NextDouble() < crossoverChance)
            {
                weights1.Add(parent2.Weights[i]);
                weights2.Add(parent1.Weights[i]);
            }
            else
            {
                weights1.Add(parent1.Weights[i]);
                weights2.Add(parent2.Weights[i]);
            }
        }
        
        return (new Genotype(weights1, 0), new Genotype(weights2, 0));
    }

    private (Genotype, Genotype) OnepointCrossover(Genotype parent1, Genotype parent2)
    {
        int weightsCount = parent1.Weights.Count;

        if (parent2.Weights.Count != weightsCount)
        {
            Debug.LogError("Crossover parents have different sizes");
            return (parent1, parent2); // Return original if sizes differ
        }

        // Choose a random crossover point
        int crossoverPoint = _rand.Next(0, weightsCount);

        List<double> weights1 = new List<double>();
        List<double> weights2 = new List<double>();

        weights1.AddRange(parent1.Weights.Take(crossoverPoint));
        weights1.AddRange(parent2.Weights.Skip(crossoverPoint));

        weights2.AddRange(parent2.Weights.Take(crossoverPoint));
        weights2.AddRange(parent1.Weights.Skip(crossoverPoint));

        return (new Genotype(weights1, 0), new Genotype(weights2, 0));
    }

    private List<Genotype> Mutate(List<Genotype> generation)
    {
        for (var i = 0; i < generation.Count; i++)
        {
            int weightsCount = generation[i].Weights.Count;
            var weights = generation[i].Weights;
            
            for (var j = 0; j < weightsCount; j++)
            {
                if (_rand.NextDouble() < mutationRate)
                {
                    weights[j] += _rand.NextDouble() * mutationAmount * 2 - mutationAmount;
                }
            }

            generation[i] = new Genotype(weights, 0);
        }

        return generation;
    }

    private List<double> InitNegOneToOne(int NIn, int NOut)
    {
        var weights = new List<double>();
        int size = NIn * NOut;

        for (var i = 0; i < size; i++)
        {
            weights.Add(_rand.NextDouble() * 2 - 1);
        }

        return weights;
    }

    private List<double> GenerateGaussianList(int size, double mean = 0.0, double stddev = 1.0)
    {
        var gaussianNumbers = new List<double>();

        for (var i = 0; i < size; i++)
        {
            gaussianNumbers.Add(RandomGaussian(mean, stddev));
        }

        return gaussianNumbers;
    }

    private double RandomGaussian(double mean = 0.0, double stddev = 1.0)
    {
        // Use the Box-Muller transform
        double u1 = 1.0 - _rand.NextDouble(); // Uniform(0,1] random number
        double u2 = 1.0 - _rand.NextDouble();
        double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        return z * stddev + mean;
    }
}
