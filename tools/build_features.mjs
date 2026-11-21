import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';

const base = path.resolve('outputs/electropv');
const rawDir = path.join(base, 'data', 'raw_csv', 'CSV_Files');
const outDir = path.join(base, 'data', 'csv_pruebas');
const zipInput = path.join(base, 'data', 'source', 'GPVS-Faults.zip');
fs.mkdirSync(outDir, { recursive: true });
const channels = ['Ipv','Vpv','Vdc','ia','ib','ic','va','vb','vc','Iabc','If','Vabc','Vf'];
const win = 1024;

async function countRows(file) {
  let count = 0;
  const rl = readline.createInterface({ input: fs.createReadStream(file), crlfDelay: Infinity });
  for await (const line of rl) if (line && !line.startsWith('Time,')) count++;
  return count;
}

function getFeatures(rows) {
  const result = [];
  for (let c=0;c<channels.length;c++) {
    let sum=0, sum2=0, max=0;
    for(const row of rows){const x=row[c];sum+=x;sum2+=x*x;max=Math.max(max,Math.abs(x));}
    const n=rows.length, mean=sum/n, rms=Math.sqrt(sum2/n), variance=Math.max(0,sum2/n-mean*mean), sd=Math.sqrt(variance);
    let m4=0;
    if(sd>1e-12) for(const row of rows){const d=row[c]-mean;m4+=d*d*d*d;}
    const kurt=sd>1e-12 ? m4/n/(variance*variance) : 0;
    result.push(mean,rms,sd,kurt,rms>1e-12?max/rms:0);
  }
  return result;
}

for (const fileName of fs.readdirSync(rawDir).filter(n => /^F[0-7][LM]\.csv$/i.test(n)).sort()) {
  const source = path.join(rawDir,fileName), total=await countRows(source), boundary=Math.floor(total/2);
  const rows=[]; let idx=0, winNum=0, output=[];
  const rl=readline.createInterface({input:fs.createReadStream(source),crlfDelay:Infinity});
  for await(const line of rl){
    if(!line||line.startsWith('Time,'))continue;
    const vals=line.split(',').slice(1).map(Number);
    if(vals.length!==13||vals.some(v=>!Number.isFinite(v))){idx++;continue;}
    rows.push(vals);idx++;
    if(rows.length===win){
      const mid=idx-rows.length/2, scenario=Number(fileName[1]), label=(scenario===0||mid<=boundary)?0:scenario;
      output.push([fileName.replace('.csv',''),fileName[2],winNum++,mid,mid<=boundary?'Antes de la falla':(scenario===0?'Operación normal':'Después de la falla'),label,...getFeatures(rows)].join(','));
      rows.length=0;
    }
  }
  if(rows.length>=win/2){
    const mid=idx-rows.length/2, scenario=Number(fileName[1]), label=(scenario===0||mid<=boundary)?0:scenario;
    output.push([fileName.replace('.csv',''),fileName[2],winNum++,mid,mid<=boundary?'Antes de la falla':(scenario===0?'Operación normal':'Después de la falla'),label,...getFeatures(rows)].join(','));
  }
  const featNames=channels.flatMap(c=>['media','rms','desv','curtosis','factor_cresta'].map(f=>`${c}_${f}`));
  const header=['archivo_origen','modo','ventana','indice_medio','fase','etiqueta',...featNames].join(',');
  fs.writeFileSync(path.join(outDir,fileName),header+'\n'+output.join('\n')+'\n','utf8');
  console.log(`${fileName}: ${total.toLocaleString()} lecturas -> ${output.length} ventanas (${(fs.statSync(path.join(outDir,fileName)).size/1024).toFixed(1)} KiB)`);
}

console.log(`ZIP fuente: ${zipInput}; features guardadas en ${outDir}`);

