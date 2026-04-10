// Import Firebase functions
import { auth, db } from "./firebase.js";
import { createUserWithEmailAndPassword, updateProfile } from "https://www.gstatic.com/firebasejs/12.3.0/firebase-auth.js";
import { doc, setDoc } from "https://www.gstatic.com/firebasejs/12.3.0/firebase-firestore.js";

export async function handleRegistration(e) {
  e.preventDefault();

  // Get form values
  const fullname = document.getElementById("fullname")?.value.trim();
  const email = document.getElementById("email")?.value.trim();
  const password = document.getElementById("password")?.value.trim();
  const confirmPassword = document.getElementById("confirmPassword")?.value.trim();

  // Debug logging
  console.log("Form submitted with:", { fullname, email, password: '***' });

  // Validation
  if (!fullname || !email || !password || !confirmPassword) {
    alert("⚠ Please fill all fields!");
    return;
  }
  if (password !== confirmPassword) {
    alert("⚠ Passwords do not match!");
    return;
  }
  if (password.length < 6) {
    alert("⚠ Password must be at least 6 characters!");
    return;
  }

  try {
    console.log("Attempting to create user...");

    // Create user in Firebase Authentication
    const userCredential = await createUserWithEmailAndPassword(auth, email, password);
    const user = userCredential.user;

    console.log("User created:", user.uid);

    // Update user profile with full name
    await updateProfile(user, { displayName: fullname });
    console.log("Profile updated");

    // Save extra info in Firestore
    await setDoc(doc(db, "users", user.uid), {
      createdAt: new Date()
    });
    console.log("User data saved to Firestore");

    // Show a non-blocking message above the form
    const successMsg = document.createElement("div");
    successMsg.className = "text-center py-4 mb-4 bg-green-700/80 rounded-xl";
    successMsg.innerHTML = `
      <div class="text-2xl mb-2">✅ Registration successful!</div>
      <div class="text-sm text-gray-200">Redirecting to login page...</div>
    `;
    const mainCard = document.querySelector("main .max-w-md");
    if (mainCard) {
      mainCard.insertBefore(successMsg, mainCard.firstChild);
    }

    // Disable the form to prevent further input
    const registerForm = document.getElementById("registerForm");
    if (registerForm) {
      registerForm.querySelectorAll("input, button").forEach(el => el.disabled = true);
      registerForm.reset();
    }

    console.log("Registration successful. Preparing to redirect to login.html in 500ms.");

    // Delay redirect to ensure Firebase operations complete
    setTimeout(() => {
      console.log("Redirecting to login.html now.");
      window.location.href = "login.html";
    }, 500);

  } catch (error) {
    if (error.code === "auth/network-request-failed") {
      alert("Network error: Please check your internet connection and try again.");
      console.error("Network error during registration:", error);
    } else {
      alert("Registration failed: " + error.message);
      console.error("Registration error:", error);
    }
  }
}