"use client";

import { Amplify } from "aws-amplify";
import { cognitoUserPoolsTokenProvider } from "aws-amplify/auth/cognito";
import { ReactNode } from "react";

const userPoolId = process.env.NEXT_PUBLIC_COGNITO_USER_POOL_ID;
const clientId = process.env.NEXT_PUBLIC_COGNITO_CLIENT_ID;
const region = process.env.NEXT_PUBLIC_COGNITO_REGION;

console.log("[AmplifyProvider] Initializing with:", {
  userPoolId: userPoolId ? "✓ set" : "✗ MISSING",
  clientId: clientId ? "✓ set" : "✗ MISSING",
  region: region || "us-east-1",
});

try {
  // Configure Cognito token provider to use localStorage BEFORE Amplify.configure
  const localStorageAdapter = {
    getItem: (key: string) => {
      try {
        if (typeof window !== "undefined") {
          return localStorage.getItem(key);
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
        }
      } catch (e) {
        console.error(`[AmplifyProvider] setItem error for "${key}":`, e);
      }
    },
    removeItem: (key: string) => {
      try {
        if (typeof window !== "undefined") {
          localStorage.removeItem(key);
        }
      } catch (e) {
        console.error(`[AmplifyProvider] removeItem error for "${key}":`, e);
      }
    },
  };

  console.log("[AmplifyProvider] Setting token provider storage adapter...");
  cognitoUserPoolsTokenProvider.setKeyValueStorage(localStorageAdapter as any);

  Amplify.configure({
    Auth: {
      Cognito: {
        userPoolId: userPoolId!,
        userPoolClientId: clientId!,
      },
    },
  });

  console.log("[AmplifyProvider] ✓ Configuration complete");
} catch (error) {
  console.error("[AmplifyProvider] ✗ Configuration failed:", error);
}

export default function AmplifyProvider({
  children,
}: {
  children: ReactNode;
}) {
  console.log("[AmplifyProvider] Rendering children");
  return <>{children}</>;
}
