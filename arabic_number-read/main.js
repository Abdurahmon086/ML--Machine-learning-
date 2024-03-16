// C:\Users\Abdurahmon\Desktop\ML (Machine learning)\arabic_number-read\main.js

// fetch kutubxonasi va TensorFlow.js kutubxonasi
import('node-fetch').then(async (nodeFetch) => {
    const tf = require('@tensorflow/tfjs');
    const fetch = nodeFetch.default;
  
    // Rasmlarni yuklash va qayta o'lchamlandirish
    const loadImage = async (path) => {
      const img = new tf.node.Image();
      await img.decode(new Uint8Array(await (await fetch(path)).arrayBuffer()));
      const canvas = tf.node.decodeImage(img.toTensor()).expandDims(0);
      return canvas;
    };
  
    // Model va natijani aniqlash
    async function loadModelAndPredict() {
      // Modelni yuklash
      const modelPath = 'path/to/your/model_js';
      const model = await tf.loadLayersModel(`file://${modelPath}/model.json`);
  
      // O'zingizning rasmingizni aniqlash
      const imgPath = 'path/to/your/image.png'; // O'zingizning rasmingizning joyi
      const img = await loadImage(imgPath);
  
      // Rasmlarni ma'lumotlarga o'zgartirish
      const input = img.toFloat().div(255);
  
      // Natijani olish
      const prediction = model.predict(input);
      const predictedDigit = prediction.argMax(1).dataSync()[0];
  
      // Natijani chiqarish
      console.log(`Aniqlangan raqam: ${predictedDigit}`);
    }
  
    // Funksiyani chaqirish
    loadModelAndPredict();
  });
  
  