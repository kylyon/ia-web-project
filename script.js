// --- Variables Globales ---
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const resultValue = document.getElementById('result-value');
const resultLabel = document.getElementById('result-label');
const modelStatus = document.getElementById('model-status');
const modelSelect = document.getElementById('model-select');
var selectedModel = "my_model"

let session;
let isDrawing = false;

// --- Initialisation du Canvas ---
function initCanvas() {
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "black";
    ctx.lineWidth = 20;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
}
initCanvas();

// --- Gestion des événements de dessin ---
const startDraw = (e) => { isDrawing = true; draw(e); };
const stopDraw = () => { isDrawing = false; ctx.beginPath(); };
const drawing = (e) => draw(e);

// Souris
canvas.addEventListener('mousedown', startDraw);
canvas.addEventListener('mousemove', drawing);
canvas.addEventListener('mouseup', stopDraw);
canvas.addEventListener('mouseout', stopDraw);

// Tactile (Mobile)
canvas.addEventListener('touchstart', (e) => { e.preventDefault(); startDraw(e.touches[0]); });
canvas.addEventListener('touchmove', (e) => { e.preventDefault(); drawing(e.touches[0]); });
canvas.addEventListener('touchend', stopDraw);

function draw(e) {
    if (!isDrawing) return;
    const rect = canvas.getBoundingClientRect();
    
    // Position relative à la zone de dessin
    const clientX = e.clientX || e.pageX;
    const clientY = e.clientY || e.pageY;
    
    const x = clientX - rect.left;
    const y = clientY - rect.top;

    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
}

// --- Bouton Effacer ---
document.getElementById('clear-btn').addEventListener('click', () => {
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    resultValue.innerText = "-";
    resultLabel.innerText = "En attente";
    resultValue.style.color = "var(--text-main)";
});

// Gestion event choix modele
modelSelect.addEventListener("change", (e) => {selectedModel = e.currentTarget.value; loadModel()})

// --- Chargement du Modèle ONNX ---
async function loadModel() {
    modelStatus.classList.remove('ready');
    resultValue.innerText = "-";
    resultLabel.innerText = "En attente";
    resultValue.style.color = "var(--text-main)";
    try {
        resultLabel.innerText = "Chargement...";
        
        // IMPORTANT: Le chemin './model.onnx' suppose que le modèle 
        // est dans le même dossier que le fichier HTML
        session = await ort.InferenceSession.create('./'+ selectedModel + '.onnx');
        
        // Mise à jour de l'UI
        modelStatus.classList.add('ready'); // Point vert
        resultLabel.innerText = "Prêt";
        console.log("Modèle chargé avec succès");
        
    } catch (e) {
        console.error(e);
        resultLabel.innerText = "Erreur Modèle";
        resultValue.innerText = "!";
        resultValue.style.color = "var(--danger)";
        alert("Erreur: Impossible de charger 'model.onnx'. Assurez-vous d'utiliser un serveur local (http://localhost...).");
    }
}

// Démarrer le chargement
loadModel();

// --- Logique de Prédiction ---
document.getElementById('predict-btn').addEventListener('click', async () => {
    if (!session) {
        alert("Le modèle n'est pas encore chargé.");
        return;
    }

    try {
        resultLabel.innerText = "Calcul...";

        // 1. Redimensionnement (280x280 -> 28x28)
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = 28;
        tempCanvas.height = 28;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.drawImage(canvas, 0, 0, 28, 28);
        
        // 2. Extraction des données
        const imgData = tempCtx.getImageData(0, 0, 28, 28);
        const float32Data = new Float32Array(28 * 28);
        
        for (let i = 0; i < 28 * 28; i++) {
            // Normalisation : on prend le canal rouge, divisé par 255.0
            float32Data[i] = ((imgData.data[i * 4] / 255.0) - 1) * -1;
        }

        // 3. Création du tenseur [1, 1, 28, 28]
        const inputTensor = new ort.Tensor('float32', float32Data, [1, 1, 28, 28]);

        // 4. Inférence (Run)
        const inputName = session.inputNames[0];
        const outputName = session.outputNames[0];
        
        const feeds = {};
        feeds[inputName] = inputTensor;
        
        const results = await session.run(feeds);
        const output = results[outputName].data;

        // 5. Recherche de l'index Max (Argmax)
        let maxProb = -Infinity;
        let predictedDigit = -1;
        for (let i = 0; i < output.length; i++) {
            if (output[i] > maxProb) {
                maxProb = output[i];
                predictedDigit = i;
            }
        }

        // 6. Affichage du résultat
        resultLabel.innerText = "Prédiction";
        resultValue.innerText = predictedDigit;
        resultValue.style.color = "var(--success)"; // Vert
        
        // Animation simple
        resultValue.animate([
            { transform: 'scale(0.8)', opacity: 0.5 },
            { transform: 'scale(1.2)', opacity: 1 },
            { transform: 'scale(1)', opacity: 1 }
        ], { duration: 300, easing: 'ease-out' });

    } catch (e) {
        console.error("Erreur durant la prédiction:", e);
        resultLabel.innerText = "Erreur";
    }
});