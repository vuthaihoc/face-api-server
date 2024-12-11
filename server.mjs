import express from 'express';
import * as faceapi from 'face-api.js';
import {loadImage, createCanvas, Image, ImageData, Canvas} from 'canvas';
import fs from 'fs';
import path from 'path';

const app = express();
const PORT = process.env.PORT || 3000;

// Monkey patch canvas
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

// Tải mô hình
async function loadModels() {
  const modelPath = path.join('.', 'weights');
  await faceapi.nets.ssdMobilenetv1.loadFromDisk(modelPath);
  await faceapi.nets.faceLandmark68Net.loadFromDisk(modelPath);
  await faceapi.nets.faceRecognitionNet.loadFromDisk(modelPath);
}

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.post('/detect-face', async (req, res) => {
  const { imagePath } = req.body; // Đường dẫn đến bức ảnh
  try {
    const img = await loadImage(imagePath);
    const detections = await faceapi.detectSingleFace(img).run();
    res.json(detections);
  } catch (error) {
    console.error('Error detecting face:', error);
    res.status(500).json({ error: 'Error detecting face' });
  }
});

loadModels().then(() => {
  app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
  });
}).catch(error => {
  console.error('Error loading models:', error);
});