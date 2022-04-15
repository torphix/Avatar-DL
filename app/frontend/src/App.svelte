<script>
	export let name;
	import { onMount } from "svelte";
	// Audio
	let media = [];
	let mediaRecorder = null;
	onMount(async () => {
		const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
		mediaRecorder = new MediaRecorder(stream);
		mediaRecorder.ondataavailable = (e) => media.push(e.data);
		mediaRecorder.onstop = function () {
			const audio = document.querySelector("audio");
			const blob = new Blob(media, { type: "audio/ogg; codecs=opus" });
			media = []; // <-- We reset our media array
			audio.src = window.URL.createObjectURL(blob);
		};
	});
	function startRecording() {
		mediaRecorder.start();
	}
	function stopRecording() {
		mediaRecorder.stop();
	}
	// Socket
	let socket = io('http://localhost:5000/audio');

</script>

<main>
	<h1>Hello {name}!</h1>
	<p>
		Visit the <a href="https://svelte.dev/tutorial">Svelte tutorial</a> to learn
		how to build Svelte apps.
	</p>

	<section>
		<audio controls />
		<button on:click={startRecording}>Record</button>
		<button on:click={stopRecording}>Stop</button>
	</section>
</main>

<style>
	main {
		text-align: center;
		padding: 1em;
		max-width: 240px;
		margin: 0 auto;
	}

	h1 {
		color: #ff3e00;
		text-transform: uppercase;
		font-size: 4em;
		font-weight: 100;
	}

	@media (min-width: 640px) {
		main {
			max-width: none;
		}
	}
</style>
