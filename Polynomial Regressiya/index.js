const fs = require('fs');
const csv = require('csv-parser');
const PolynomialRegression = require('ml-regression').PolynomialRegression;

// Read the CSV file
const data = [];
fs.createReadStream('Salary_Data.csv')
    .pipe(csv())
    .on('data', (row) => {
        data.push({ x: parseFloat(row['YearsExperience']), y: parseFloat(row['Salary']) });
    })
    .on('end', () => {
        const x = data.map(item => item.x);
        const y = data.map(item => item.y);
        const degree = 5;

        const regression = new PolynomialRegression(x, y, degree);

        console.log(regression.predict(8)); // Example prediction for 8 years of experience

        console.log(regression.coefficients);

        console.log(regression.toString(3));

        console.log(regression.toLaTeX());
    });

