import { auth, db } from "./firebase.js";
import {
  createUserWithEmailAndPassword,
  updateProfile
} from "https://www.gstatic.com/firebasejs/12.3.0/firebase-auth.js";
import { doc, setDoc } from "https://www.gstatic.com/firebasejs/12.3.0/firebase-firestore.js";

// Readable Firebase error messages
function readableError(code) {
  const m = {
    "auth/email-already-in-use":   "An account with this email already exists. Try logging in instead.",
    "auth/invalid-email":          "Please enter a valid email address.",
    "auth/weak-password":          "Password must be at least 6 characters.",
    "auth/operation-not-allowed":  "Email/password sign-up is not enabled. Enable it in Firebase Console → Authentication → Sign-in method.",
    "auth/network-request-failed": "Network error. Please check your internet connection.",
    "auth/too-many-requests":      "Too many attempts. Please wait and try again.",
  };
  return m[code] || null;
}

function showBanner(msg, type = "error") {
  let el = document.getElementById("regBanner");
  if (!el) {
    el = document.createElement("div");
    el.id = "regBanner";
    const form = document.getElementById("registerForm");
    form.parentNode.insertBefore(el, form);
  }
  el.className = "mb-4 rounded-xl px-4 py-3 text-sm text-center " + (
    type === "success"
      ? "bg-green-800/50 border border-green-600/50 text-green-200"
      : "bg-red-900/50 border border-red-700/50 text-red-200"
  );
  el.textContent = msg;
}

export async function handleRegistration(e) {
  e.preventDefault();

  const fullname        = document.getElementById("fullname")?.value.trim();
  const email           = document.getElementById("email")?.value.trim();
  const password        = document.getElementById("password")?.value;
  const confirmPassword = document.getElementById("confirmPassword")?.value;
  const submitBtn       = document.querySelector("#registerForm button[type='submit']");

  // Client-side validation
  if (!fullname || !email || !password || !confirmPassword) {
    showBanner("⚠️ Please fill in all fields.");
    return;
  }
  if (password !== confirmPassword) {
    showBanner("⚠️ Passwords do not match.");
    return;
  }
  if (password.length < 6) {
    showBanner("⚠️ Password must be at least 6 characters.");
    return;
  }

  submitBtn.disabled = true;
  submitBtn.textContent = "Creating account…";

  try {
    // ── Step 1: Create Firebase Auth user ─────────────────────────────────────
    const { user } = await createUserWithEmailAndPassword(auth, email, password);

    // ── Step 2: Set display name (non-blocking) ───────────────────────────────
    try {
      await updateProfile(user, { displayName: fullname });
    } catch (profileErr) {
      console.warn("[EduBot] Profile update skipped:", profileErr.message);
    }

    // ── Step 3: Save to Firestore (non-blocking — don't fail registration) ─────
    try {
      await setDoc(doc(db, "users", user.uid), {
        name:      fullname,
        email:     email,
        createdAt: new Date().toISOString()
      });
    } catch (fsErr) {
      console.warn("[EduBot] Firestore save skipped:", fsErr.code, fsErr.message);
    }

    // ── Success ───────────────────────────────────────────────────────────────
    showBanner("✅ Account created! Redirecting to login…", "success");
    document.getElementById("registerForm")
      .querySelectorAll("input,button")
      .forEach(el => el.disabled = true);

    setTimeout(() => { window.location.href = "login.html"; }, 1000);

  } catch (err) {
    console.error("[EduBot] Registration error:", err.code, err.message);
    showBanner(readableError(err.code) || err.message);
    submitBtn.disabled = false;
    submitBtn.textContent = "Register";
  }
}
