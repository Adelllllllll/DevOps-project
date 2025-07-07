document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("prediction-form");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const text = document.getElementById("text").value;
    const rating = parseFloat(document.getElementById("rating").value);
    const category = document.getElementById("category").value;

    const payload = {
      text: text,
      rating: rating,
      product_category: category
    };

    try {
      const res = await fetch("https://devops-project-production.up.railway.app/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        throw new Error("Erreur lors de la prédiction.");
      }

      const data = await res.json();
      document.getElementById("result").textContent = "Prediction: " + data.prediction;
    } catch (error) {
      document.getElementById("result").textContent = "Erreur: " + error.message;
    }
  });
});
