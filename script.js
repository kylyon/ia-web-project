ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

// --- 1. Variables Globales ---
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const resultDiv = document.getElementById('result');
let session; // Session ONNX
let isDrawing = false;

// --- 2. Initialisation du Canvas ---
ctx.fillStyle = "white";
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.strokeStyle = "black"; // Encre blanche
ctx.lineWidth = 15;        // Épaisseur du trait
ctx.lineCap = "round";     // Bords arrondis

// --- 3. Gestion de la souris pour dessiner ---
canvas.addEventListener('mousedown', (e) => { isDrawing = true; draw(e); });
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', () => isDrawing = false);
canvas.addEventListener('mouseout', () => isDrawing = false);

// Support tactile basique
canvas.addEventListener('touchstart', (e) => { isDrawing = true; draw(e.touches[0]); e.preventDefault(); });
canvas.addEventListener('touchmove', (e) => { draw(e.touches[0]); e.preventDefault(); });
canvas.addEventListener('touchend', () => isDrawing = false);

function draw(e) {
    if (!isDrawing) return;
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX || e.pageX) - rect.left;
    const y = (e.clientY || e.pageY) - rect.top;

    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
}

// Réinitialiser le tracé (nécessaire à chaque nouveau trait)
canvas.addEventListener('mousedown', () => ctx.beginPath());
canvas.addEventListener('touchstart', () => ctx.beginPath());

// --- 4. Bouton Effacer ---
document.getElementById('clear-btn').addEventListener('click', () => {
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    resultDiv.innerText = "Dessinez un chiffre...";
});

// --- 5. Chargement du Modèle ONNX ---
async function loadModel() {
    try {
        // Création de la session d'inférence
        // Assurez-vous d'avoir 'model.onnx' dans le même dossier
        session = await ort.InferenceSession.create('./model.onnx');
        resultDiv.innerText = "Modèle chargé ! Dessinez.";
        resultDiv.classList.remove('loading');
    } catch (e) {
        console.error(e);
        resultDiv.innerHTML = `<span class="error">Erreur: Impossible de charger 'model.onnx'.<br>Vérifiez que le fichier est présent et que vous utilisez un serveur local.</span>`;
    }
}

loadModel();

// --- 6. Logique de Prédiction ---
document.getElementById('predict-btn').addEventListener('click', async () => {
    if (!session) {
        alert("Le modèle n'est pas encore chargé !");
        return;
    }

    // A. Prétraitement de l'image
    // On crée un petit canvas temporaire pour redimensionner l'image à 28x28
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempCtx = tempCanvas.getContext('2d');
    
    // On dessine l'image de 280x280 vers 28x28
    tempCtx.drawImage(canvas, 0, 0, 28, 28);
    
    const imageData = tempCtx.getImageData(0, 0, 28, 28);
    const data = imageData.data; // Tableau RGBA [r, g, b, a, r, g, b, a...]
    
    // B. Conversion en Tensor Float32 [1, 1, 28, 28]
    // On ne garde que le canal Rouge (car c'est en noir et blanc) et on normalise entre 0 et 1
    const float32Data = new Float32Array(28 * 28);
    for (let i = 0; i < 28 * 28; i++) {
        // data[i*4] est le canal Rouge. 
        // On divise par 255.0 pour avoir des valeurs entre 0.0 et 1.0
        float32Data[i] = ((data[i * 4] / 255.0) - 1) * - 1;
    }

    // Création du tenseur ONNX
    const inputTensor = new ort.Tensor('float32', float32Data, [1, 1, 28, 28]);

    // C. Inférence
    try {
        // On récupère le nom de l'entrée du modèle dynamiquement
        const inputName = session.inputNames[0];
        const feeds = {};
        feeds[inputName] = inputTensor;

        // Exécution
        const results = await session.run(feeds);
        
        // On récupère la sortie (généralement nommée 'Output3', 'Plus214_Output_0', etc.)
        const outputName = session.outputNames[0];
        const outputTensor = results[outputName];
        const outputData = outputTensor.data;

        console.log(outputData)

        // D. Trouver l'index de la valeur maximale (Argmax)
        let maxProb = -Infinity;
        let predictedDigit = -1;

        for (let i = 0; i < outputData.length; i++) {
            if (outputData[i] > maxProb) {
                maxProb = outputData[i];
                predictedDigit = i;
            }
        }

        resultDiv.innerText = `Prédiction : ${predictedDigit}`;
        
    } catch (e) {
        console.error(e);
        resultDiv.innerText = "Erreur lors de la prédiction.";
    }
});