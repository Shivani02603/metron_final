"use client";

import { Amplify } from "aws-amplify";
import { cognitoUserPoolsTokenProvider } from "aws-amplify/auth/cognito";

const userPoolId = process.env.NEXT_PUBLIC_COGNITO_USER_POOL_ID;
const clientId = process.env.NEXT_PUBLIC_COGNITO_CLIENT_ID;
const region = process.env.NEXT_PUBLIC_COGNITO_REGION;

console.log("[AmplifyProvider] Initializing with:", {
  userPoolId: userPoolId ? "✓ set" : "✗ MISSING",
  clientId: clientId ? "✓ set" : "✗ MISSING",
  region: region || "us-east-1",
});

// Configure Cognito token provider to use localStorage BEFORE Amplify.configure
const localStorageAdapter = {
  getItem: (key: string) => {
    try {
      if (typeof window !== "undefined") {
        const value = localStorage.getItem(key);
        console.log(`[AmplifyProvider] getItem("${key}"):`, value ? `✓ ${value.substring(0, 20)}...` : "null");
        return value;
      }
      return null;
    } catch (e) {
      console.error(`[AmplifyProvider] getItem error for "${key}":`, e);
      return null;
    }
  },
  setItem: (key: string, value: string) => {
    try {
      if (typeof window !== "undefined") {
        localStorage.setItem(key, value);
        console.log(`[AmplifyProvider] setItem("${key}"):`, `✓ stored ${value.substring(0, 20)}...`);
      }
    } catch (e) {
      console.error(`[AmplifyProvider] setItem error for "${key}":`, e);
    }
  },
  removeItem: (key: string) => {
    try {
      if (typeof window !== "undefined") {
        localStorage.removeItem(key);
        console.log(`[AmplifyProvider] removeItem("${key}"):`, "✓ removed");
      }
    } catch (e) {
      console.error(`[AmplifyProvider] removeItem error for "${key}":`, e);
    }
  },
};

// Set storage BEFORE Amplify.configure
console.log("[AmplifyProvider] Setting token provider storage adapter...");
cognitoUserPoolsTokenProvider.setKeyValueStorage(localStorageAdapter as any);

Amplify.configure({
  Auth: {
    Cognito: {
      userPoolId: userPoolId!,
      userPoolClientId: clientId!,
      region: region || "us-east-1",
    },
  },
});

console.log("[AmplifyProvider] Configuration complete with localStorage storage");

export default function AmplifyProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  return <>{children}</>;
}
