import { fetchAuthSession } from "aws-amplify/auth";
 
export async function authFetch(
  input: RequestInfo | URL,
  init?: RequestInit
): Promise<Response> {
  let token = "";
  try {
    const session = await fetchAuthSession();
    token = session.tokens?.idToken?.toString() ?? "";
  } catch {
    // not signed in — backend will return 401
  }
 
  const headers = new Headers(init?.headers);
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }
 
  return fetch(input, { ...init, headers });
}
 
 