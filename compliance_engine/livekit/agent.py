"""
agent.py
--------
LiveKit real-time audio streaming agent for compliance monitoring.

Integrates with LiveKit to receive real-time audio streams, transcribe them,
segment the transcripts, and process chunks through the compliance engine.
Maintains per-user session state for continuous monitoring.
"""

import asyncio
import logging
import tempfile
import time
from collections import defaultdict
from typing import Dict, Optional

from livekit import Room, RoomEvent, rtc

from compliance_engine.audio.transcriber import AudioTranscriber
from compliance_engine.compliance_engine import ComplianceEngine
from compliance_engine.streaming.processor import StreamingProcessor
from compliance_engine.streaming.segmenter import simulate_segmentation

logger = logging.getLogger(__name__)


class LiveKitAgent:
    """
    Real-time audio streaming agent using LiveKit for compliance monitoring.

    Connects to a LiveKit room, subscribes to audio tracks, and processes
    real-time audio chunks through transcription, segmentation, and compliance analysis.
    """

    def __init__(
        self,
        room_url: str,
        token: str,
        domain: str,
        compliance_engine: Optional[ComplianceEngine] = None,
        audio_provider: str = "whisper",
        chunk_duration: float = 3.0,  # seconds
        sample_rate: int = 16000,
    ):
        """
        Initialize the LiveKit agent.

        Parameters
        ----------
        room_url : str
            LiveKit room URL
        token : str
            LiveKit access token
        domain : str
            Regulatory domain for compliance analysis
        compliance_engine : Optional[ComplianceEngine]
            Compliance engine instance. If None, creates a default one.
        audio_provider : str
            Audio transcription provider ("whisper" or "sarvam")
        chunk_duration : float
            Duration in seconds to accumulate audio before transcription
        sample_rate : int
            Audio sample rate for processing
        """
        self.room_url = room_url
        self.token = token
        self.domain = domain
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate

        # Initialize components
        self.engine = compliance_engine or ComplianceEngine()
        self.transcriber = AudioTranscriber(provider=audio_provider)

        # Session state: participant_id -> StreamingProcessor
        self.sessions: Dict[str, StreamingProcessor] = {}

        # Audio buffers: participant_id -> list of audio frames
        self.audio_buffers: Dict[str, list] = defaultdict(list)

        # Timestamps for chunking: participant_id -> last_process_time
        self.last_process_times: Dict[str, float] = {}

        # Room and connection
        self.room: Optional[Room] = None

    async def connect(self) -> None:
        """Connect to the LiveKit room."""
        self.room = Room()
        self.room.on(RoomEvent.TrackSubscribed, self._on_track_subscribed)
        self.room.on(RoomEvent.ParticipantDisconnected, self._on_participant_disconnected)

        logger.info("Connecting to LiveKit room: %s", self.room_url)
        await self.room.connect(self.room_url, self.token)
        logger.info("Connected to LiveKit room successfully")

    async def disconnect(self) -> None:
        """Disconnect from the LiveKit room."""
        if self.room:
            await self.room.disconnect()
            logger.info("Disconnected from LiveKit room")

    def _on_track_subscribed(self, track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant) -> None:
        """Handle track subscription."""
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            logger.info("Subscribed to audio track for participant: %s", participant.identity)
            track.on(rtc.TrackEvent.FrameReceived, lambda frame: self._on_audio_frame(frame, participant.identity))

    def _on_participant_disconnected(self, participant: rtc.RemoteParticipant) -> None:
        """Handle participant disconnection."""
        participant_id = participant.identity
        if participant_id in self.sessions:
            logger.info("Cleaning up session for disconnected participant: %s", participant_id)
            del self.sessions[participant_id]
            del self.audio_buffers[participant_id]
            del self.last_process_times[participant_id]

    def _on_audio_frame(self, frame: rtc.AudioFrame, participant_id: str) -> None:
        """Handle incoming audio frame."""
        # Buffer the audio frame
        self.audio_buffers[participant_id].append(frame.data)

        # Check if we should process the chunk
        current_time = time.time()
        last_time = self.last_process_times.get(participant_id, 0)

        if current_time - last_time >= self.chunk_duration:
            asyncio.create_task(self._process_audio_chunk(participant_id))
            self.last_process_times[participant_id] = current_time

    async def _process_audio_chunk(self, participant_id: str) -> None:
        """Process accumulated audio chunk for a participant."""
        if not self.audio_buffers[participant_id]:
            return

        # Combine audio frames into a single chunk
        audio_data = b"".join(self.audio_buffers[participant_id])
        self.audio_buffers[participant_id].clear()

        if len(audio_data) == 0:
            return

        logger.info("Processing audio chunk for participant %s (size: %d bytes)", participant_id, len(audio_data))

        try:
            # Save audio data to temporary file for transcription
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                # Write WAV header (simplified - assumes 16-bit PCM)
                import wave
                with wave.open(temp_file.name, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(self.sample_rate)
                    wav_file.writeframes(audio_data)

                temp_path = temp_file.name

            # Transcribe the audio chunk
            transcription_result = self.transcriber.transcribe(temp_path)
            transcript_text = transcription_result.get("text", "").strip()

            # Clean up temp file
            import os
            os.unlink(temp_path)

            if not transcript_text:
                logger.debug("No transcription text for participant %s", participant_id)
                return

            logger.info("Transcribed chunk for participant %s: '%s'", participant_id, transcript_text[:100])

            # Segment the transcription
            segments = simulate_segmentation(transcript_text)

            # Get or create session processor
            if participant_id not in self.sessions:
                self.sessions[participant_id] = StreamingProcessor(self.engine, self.domain)
                logger.info("Created new session for participant: %s", participant_id)

            processor = self.sessions[participant_id]

            # Process each segment
            for segment in segments:
                result = processor.process_chunk(segment)

                if result:
                    logger.info(
                        "Compliance check for participant %s: violation=%s, risk=%s, confidence=%.2f",
                        participant_id,
                        result.violation,
                        result.risk_level,
                        result.confidence
                    )

                    if result.violation:
                        logger.warning(
                            "VIOLATION DETECTED for participant %s: %s",
                            participant_id,
                            result.reason
                        )

        except Exception as exc:
            logger.error("Error processing audio chunk for participant %s: %s", participant_id, exc)

    async def run(self) -> None:
        """Run the agent indefinitely."""
        await self.connect()

        try:
            # Keep the connection alive
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down LiveKit agent...")
        finally:
            await self.disconnect()


# Convenience function for running the agent
async def run_livekit_agent(
    room_url: str,
    token: str,
    domain: str,
    audio_provider: str = "whisper",
    chunk_duration: float = 3.0,
) -> None:
    """
    Run the LiveKit compliance agent.

    Parameters
    ----------
    room_url : str
        LiveKit room URL
    token : str
        LiveKit access token
    domain : str
        Regulatory domain
    audio_provider : str
        Audio transcription provider
    chunk_duration : float
        Audio chunk duration in seconds
    """
    agent = LiveKitAgent(
        room_url=room_url,
        token=token,
        domain=domain,
        audio_provider=audio_provider,
        chunk_duration=chunk_duration,
    )

    await agent.run()