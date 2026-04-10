// firebase.js
import { initializeApp } from "https://www.gstatic.com/firebasejs/12.3.0/firebase-app.js";
import { getAuth } from "https://www.gstatic.com/firebasejs/12.3.0/firebase-auth.js";
import { getFirestore } from "https://www.gstatic.com/firebasejs/12.3.0/firebase-firestore.js";

// Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyBCnDx3rleO4g-euDafroqmxYHrxI2-0sU",
  authDomain: "fyp056.firebaseapp.com",
  projectId: "fyp056",
  storageBucket: "fyp056.firebasestorage.app",
  messagingSenderId: "147085120681",
  appId: "1:147085120681:web:d461246ca6617d046c05b8"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const db = getFirestore(app);

export { auth, db };