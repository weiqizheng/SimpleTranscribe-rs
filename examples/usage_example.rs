use simple_transcribe_rs::model_handler;
use simple_transcribe_rs::transcriber;

#[tokio::main]
async fn main() {
    let m = model_handler::ModelHandler::new("tiny", "models/").await;
    let trans = transcriber::Transcriber::new(m);
    let result = trans.transcribe("src/test_data/test.mp3", None).unwrap();

    for segment in result.get_segments() {
        let text = segment.get_text();
        let start = segment.get_start_timestamp();
        let end = segment.get_end_timestamp();
        println!("start[{}]-end[{}] {}", start, end, text);
    }
}
