const sharp = require('sharp');
const fs = require('fs');
const path = require('path');

/**
 * Fügt einem Bild stilisierten, zentrierten Text mit einem verfeinerten Neon-Effekt hinzu.
 * @param {string} text Der Text, der hinzugefügt werden soll.
 * @param {string} inputImagePath Pfad zum Eingabebild.
 * @param {string} outputImagePath Pfad zum Ausgabebild.
 * @param {string} fontPath Pfad zur .ttf Schriftart-Datei.
 */
async function addStyledText(text, inputImagePath, outputImagePath, fontPath) {
  try {
    // 1. Schriftart-Datei lesen und in Base64 umwandeln
    const fontBuffer = fs.readFileSync(fontPath);
    const fontBase64 = fontBuffer.toString('base64');
    const fontDataUri = `data:font/ttf;base64,${fontBase64}`;
    const fontFamily = "CustomFont";

    // 2. Bild-Metadaten auslesen
    const metadata = await sharp(inputImagePath).metadata();
    const { width, height } = metadata;

    // 3. Dynamisches SVG mit verbessertem Neon-Effekt erstellen
    const fontSize = Math.floor(width / 11); // Kleinere, dezentere Schriftgröße
    const neonColor = "#00d9ff"; // Lebhaftes Blau für den Neon-Effekt
    const textColor = "#ffffff"; // Heller Kern des Textes

    const svgText = `
    <svg width="${width}" height="${height}">
      <defs>
        <filter id="amazing-glow" x="-50%" y="-50%" width="200%" height="200%">
          <!-- Subtiler Schatten für Tiefe -->
          <feDropShadow dx="2" dy="2" stdDeviation="2" flood-color="#000000" flood-opacity="0.5" />
          <!-- Haupt-Glow -->
          <feGaussianBlur in="SourceAlpha" stdDeviation="8" result="blur" />
          <feFlood flood-color="${neonColor}" result="color" />
          <feComposite in="color" in2="blur" operator="in" result="glow" />
          <!-- Innerer, hellerer Glow -->
          <feGaussianBlur in="SourceAlpha" stdDeviation="4" result="inner-blur" />
          <feFlood flood-color="${textColor}" result="inner-color" />
          <feComposite in="inner-color" in2="inner-blur" operator="in" result="inner-glow" />
          <!-- Alles zusammenfügen -->
          <feMerge>
            <feMergeNode in="glow" />
            <feMergeNode in="inner-glow" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>
      <style>
        @font-face {
          font-family: "${fontFamily}";
          src: url("${fontDataUri}");
        }
        .title {
          fill: ${textColor};
          font-size: ${fontSize}px;
          font-family: "${fontFamily}";
          text-anchor: middle;
        }
      </style>
      <text x="50%" y="50%" dominant-baseline="middle" class="title" filter="url(#amazing-glow)">${text}</text>
    </svg>
    `;

    // 4. SVG als Buffer erstellen
    const svgBuffer = Buffer.from(svgText);

    // 5. Bild bearbeiten: SVG über das Originalbild legen
    await sharp(inputImagePath)
      .composite([{
        input: svgBuffer,
        top: 0,
        left: 0,
      }])
      .toFile(outputImagePath);

    console.log(`Text erfolgreich auf Originalbild hinzugefügt: ${outputImagePath}`);

  } catch (error) {
    console.error("Ein Fehler ist aufgetreten:", error);
  }
}

// --- Beispielaufruf ---
if (require.main === module) {
  const inputText = "Code is Poetry.";
  const inputImage = "img1.png"; // NEU: Verwende das Originalbild
  const outputImage = "temp_text_added.png"; // NEU: Speichere als temporäre Datei
  const fontFile = "Montserrat-Bold.ttf";

  // Prüfen, ob die Eingabedateien existieren
  if (!fs.existsSync(inputImage)) {
    console.error(`Fehler: Eingabebild nicht gefunden unter '${inputImage}'.`);
  } else if (!fs.existsSync(fontFile)) {
    console.error(`Fehler: Schriftart-Datei nicht gefunden unter '${fontFile}'.`);
  } else {
    addStyledText(inputText, inputImage, outputImage, fontFile);
  }
}
