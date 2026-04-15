// scripts.js
// Set current year in footer
document.getElementById("year").textContent = new Date().getFullYear();

// Very simple form handler – replace with real backend / Formspree / etc.
const form = document.getElementById("beta-form");
const message = document.getElementById("form-message");

form.addEventListener("submit", function (e) {
    e.preventDefault();
    const email = document.getElementById("email").value.trim();

    if (!email) {
        message.textContent = "Please enter a valid email.";
        message.className = "form-note form-error";
        return;
    }

    // TODO: send to your email service or backend
    console.log("Captured email:", email);

    message.textContent = "Message submitted, Thank you!";
    message.className = "form-note form-success";
    form.reset();
});