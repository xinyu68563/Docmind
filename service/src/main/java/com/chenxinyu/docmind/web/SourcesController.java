package com.chenxinyu.docmind.web;
import java.util.Map;
import com.chenxinyu.docmind.worker.WorkerClient;
import org.springframework.web.bind.annotation.*;
import org.springframework.http.*;
import jakarta.servlet.http.HttpServletRequest;
import java.io.IOException;
import com.chenxinyu.docmind.worker.WorkerException;

@RestController
@RequestMapping("/api")
public class SourcesController {
    private final WorkerClient worker;
    public SourcesController(WorkerClient worker){this.worker=worker;}
    @GetMapping("/documents") public Map<String,Object> documents(){return worker.documents();}
    @GetMapping("/imports/{id}") public Map<String,Object> importStatus(@PathVariable String id){return worker.importStatus(id);}
    @PostMapping(path="/imports",consumes="application/pdf")
    public ResponseEntity<Map<String,Object>> upload(HttpServletRequest request,
            @RequestHeader("X-File-Name") String name) throws IOException {
        int max=20*1024*1024;
        if(request.getContentLengthLong()>max)return ResponseEntity.status(413).body(Map.of("message","PDF 最大支持 20 MB"));
        byte[] bytes=request.getInputStream().readNBytes(max+1);
        if(bytes.length>max)return ResponseEntity.status(413).body(Map.of("message","PDF 最大支持 20 MB"));
        String decoded=java.net.URLDecoder.decode(name,java.nio.charset.StandardCharsets.UTF_8);
        if(!decoded.toLowerCase(java.util.Locale.ROOT).endsWith(".pdf")||bytes.length==0)
            throw new WorkerException.Rejected("请选择非空 PDF 文件",null);
        var record=worker.upload(bytes,decoded);
        return ResponseEntity.status(Boolean.TRUE.equals(record.get("duplicate"))?200:202).body(record);
    }
    @GetMapping("/chunks/{id}") public Map<String,Object> chunk(@PathVariable String id){return worker.chunk(id);}
    @GetMapping("/documents/{id}/pages/{page}") public ResponseEntity<byte[]> page(@PathVariable String id,@PathVariable int page){
        if(page<1||page>200)throw new WorkerException.Rejected("页码必须在1到200之间",null);
        var image=worker.page(id,page);
        return ResponseEntity.ok().contentType(MediaType.IMAGE_PNG).header("X-PDF-Pages",String.valueOf(image.pages()))
                .header("Cache-Control","no-store").header("X-Content-Type-Options","nosniff").body(image.body());
    }
    @GetMapping("/documents/{id}/pdf") public ResponseEntity<byte[]> pdf(@PathVariable String id){
        return ResponseEntity.ok().contentType(MediaType.APPLICATION_PDF)
                .header("X-Content-Type-Options","nosniff").body(worker.pdf(id));
    }
}
