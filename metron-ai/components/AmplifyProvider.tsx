"use client";

import { Amplify } from "aws-amplify";
import { cognitoUserPoolsTokenProvider } from "aws-amplify/auth/cognito";
import { ReactNode, useEffect } from "react";

const userPoolId = process.env.NEXT_PUBLIC_COGNITO_USER_POOL_ID;
const clientId = process.env.NEXT_PUBLIC_COGNITO_CLIENT_ID;

console.log("[AmplifyProvider] Initializing with:", {
  userPoolId: userPoolId ? "✓ set" : "✗ MISSING",
  clientId: clientId ? "✓ set" : "✗ MISSING",
});

try {
  // Configure Cognito token provider to use localStorage BEFORE Amplify.configure
  const localStorageAdapter = {
    getItem: (key: string) => {
      try {
        if (typeof window !== "undefined") {
          const value = localStorage.getItem(key);
          console.log(`[AmplifyProvider.getItem] "${key}" =>`, value ? `${value.substring(0, 30)}...` : "null");
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
          console.log(`[AmplifyProvider.setItem] "${key}" => ${value.substring(0, 30)}...`);
          localStorage.setItem(key, value);
          // Verify it was actually set
          const verify = localStorage.getItem(key);
          console.log(`[AmplifyProvider.setItem] ✓ Verified "${key}" is in localStorage`);
        }
      } catch (e) {
        console.error(`[AmplifyProvider] setItem error for "${key}":`, e);
      }
    },
    removeItem: (key: string) => {
      try {
        if (typeof window !== "undefined") {
          console.log(`[AmplifyProvider.removeItem] "${key}"`);
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
  useEffect(() => {
    console.log("[AmplifyProvider] Component mounted, checking initial localStorage state:");
    const keys = Object.keys(localStorage);
    console.log("[AmplifyProvider] localStorage keys:", keys);
    keys.forEach(key => {
      if (key.includes('token') || key.includes('idToken') || key.includes('accessToken')) {
        const value = localStorage.getItem(key);
        console.log(`  ${key}: ${value ? `${value.substring(0, 50)}...` : 'null'}`);
      }
    });
  }, []);

  return <>{children}</>;
}

