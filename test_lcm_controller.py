import lcm
import json
import time

# Channels
MANI_CMD_CHANNEL = "mani/cmd"
MANI_CALLBACK_CHANNEL = "mani/callback"

class SimpleLcmTester:
    def __init__(self):
        # Use same LCM URL as the node
        self.lc = lcm.LCM("udpm://239.255.76.67:50000?ttl=1")
        self.lc.subscribe(MANI_CALLBACK_CHANNEL, self.on_callback)
        print(f"[Tester] Subscribed to {MANI_CALLBACK_CHANNEL}")
        
    def on_callback(self, channel, data):
        try:
            msg = json.loads(data.decode('utf-8'))
            print(f"\n[Tester] RESPONSE << ID: {msg['id']}, Kind: {msg['kind']}, Obj: {msg['obj']}")
            self.response_received = True
        except Exception as e:
            print(f"[Tester] Error parsing response: {e}")

    def send_grasp(self, task_id, prompt):
        msg = {"id": str(task_id), "kind": 1, "obj": prompt}
        self.lc.publish(MANI_CMD_CHANNEL, json.dumps(msg).encode('utf-8'))
        print(f"[Tester] SEND >> Kind: 1 (Grasp), Obj: {prompt}, ID: {task_id}")

    def send_release(self, task_id):
        msg = {"id": str(task_id), "kind": 2, "obj": "release_cmd"}
        self.lc.publish(MANI_CMD_CHANNEL, json.dumps(msg).encode('utf-8'))
        print(f"[Tester] SEND >> Kind: 2 (Release), ID: {task_id}")

    def run(self):
        try:
            while True:
                print("\nOptions:")
                print("1. Send Grasp Command")
                print("2. Send Release Command")
                print("q. Quit")
                
                choice = input("Select [1/2/q]: ").strip()
                
                if choice == '1':
                    prompt = input("Enter prompt (e.g. 'purple cup'): ").strip() or "purple cup"
                    task_id = int(time.time()) # Simple unique ID
                    self.send_grasp(task_id, prompt)
                elif choice == '2':
                    task_id = int(time.time())
                    self.send_release(task_id)
                elif choice == 'q':
                    break
                else:
                    print("Invalid choice.")
                    continue
                
                # Wait for response with timeout
                print("[Tester] Waiting for response...", end="", flush=True)
                self.response_received = False
                start_wait = time.time()
                while time.time() - start_wait < 80: # 80s timeout
                    self.lc.handle_timeout(100)
                    if self.response_received:
                        break
        except KeyboardInterrupt:
            print("\nExiting...")

if __name__ == "__main__":
    tester = SimpleLcmTester()
    tester.run()
