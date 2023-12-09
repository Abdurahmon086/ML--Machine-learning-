class GeneticAlgorithm {
    constructor(target, populationSize, mutationRate) {
      this.target = target;
      this.populationSize = populationSize;
      this.mutationRate = mutationRate;
      this.population = [];
  
      // Barcha birliklarni yaratish
      for (let i = 0; i < this.populationSize; i++) {
        this.population.push(this.generateRandomDNA());
      }
    }
    generateRandomDNA() {
      let randomDNA = '';
      for (let i = 0; i < this.target.length; i++) {
        randomDNA += this.getRandomChar();
      }
      return randomDNA;
    }
    getRandomChar() {
      // Har bir belgini uni kodini olish orqali generatsiya qilamiz
      let charSet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()_-+={}[]|<>,.?/:; 0123456789';
      let charIndex = Math.floor(Math.random() * charSet.length);
      return charSet.charAt(charIndex);
    }
  
    calculateFitness(dna) {
      let fitness = 0;
      for (let i = 0; i < dna.length; i++) {
        if (dna[i] === this.target[i]) {
          fitness++;
        }
      }
      return fitness;
    }
    crossover(parentA, parentB) {
      let midpoint = Math.floor(Math.random() * this.target.length);
      let childDNA = parentA.substring(0, midpoint) + parentB.substring(midpoint);
      return childDNA;
    }
    mutate(dna) {
      let mutatedDNA = '';
      for (let i = 0; i < dna.length; i++) {
        if (Math.random() < this.mutationRate) {
          mutatedDNA += this.getRandomChar();
        } else {
          mutatedDNA += dna[i];
        }
      }
      return mutatedDNA;
    }
    evolve() {
      let newPopulation = [];
      // Elitism: eng yaxshi (eng yuqori fitnesga ega) birlikni saqlash
      let bestUnit = this.population.reduce((a, b) => (this.calculateFitness(a) > this.calculateFitness(b) ? a : b));
      newPopulation.push(bestUnit);
      // Qolgan birliklarni crossover va mutate qilish orqali yaratish
      for (let i = 1; i < this.populationSize; i++) {
        let parentA = this.population[Math.floor(Math.random() * this.populationSize)];
        let parentB = this.population[Math.floor(Math.random() * this.populationSize)];
        let childDNA = this.crossover(parentA, parentB);
        childDNA = this.mutate(childDNA);
        newPopulation.push(childDNA);
      }
      this.population = newPopulation;
    }
    findBestUnit() {
      return this.population.reduce((a, b) => (this.calculateFitness(a) > this.calculateFitness(b) ? a : b));
    }
    run(iterations) {
      for (let i = 0; i < iterations; i++) {
        this.evolve();
        let bestUnit = this.findBestUnit();
        let fitness = this.calculateFitness(bestUnit);
        console.log(`Iteration ${i + 1}: ${bestUnit} (Fitness: ${fitness})`);
  
        // Agar eng yaxshi unit maqsad so'zi bilan bir xil bo'lsa, to'xtatamiz
        if (fitness === this.target.length && bestUnit === this.target) {
          console.log(`Bingo! Topilgan so'z: ${bestUnit}`);
          break;
        }
      }
    }
  }
  let targetWord = 'Javas Cript';
  let geneticAlgorithm = new GeneticAlgorithm(targetWord, 100, 0.01);
  geneticAlgorithm.run(10000);
  