let RandomForestClassifier = require('random-forest-classifier').RandomForestClassifier;

let data = [
    {
        "length": 5.1,
        "width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
        "species": "setosa"
    },
    {
        "length": 6.5,
        "width": 3,
        "petal_length": 5.2,
        "petal_width": 2,
        "species": "virginica"
    },
    {
        "length": 6.6,
        "width": 3,
        "petal_length": 4.4,
        "petal_width": 1.4,
        "species": "versicolor"
    },
    {
        "length": 6.0,
        "width": 4,
        "petal_length": 10.5,
        "petal_width": 2,
        "species": "redColor"
    }
];

let testdata = [{
    "length": 6.3,
    "width": 2.5,
    "petal_length": 5,
    "petal_width": 1.9,
},
{
    "length": 4.7,
    "width": 3.2,
    "petal_length": 1.3,
    "petal_width": 0.3,
}
];

let rf = new RandomForestClassifier({
    n_estimators: 10
});

rf.fit(data, null, "species", function (err, trees) {
    let pred = rf.predict(testdata, trees);

    console.log(pred);
});